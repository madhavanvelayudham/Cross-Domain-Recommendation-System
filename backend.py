import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import uvicorn
from fastapi import FastAPI, File, UploadFile
from emotion_from_text import extract_emotion_scores
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rerank import rerank_recommendations
from song_emotion import build_song_emotion_map
from speech_to_text import transcribe_audio
from text_to_speech import speak_text
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer
# -----------------------------------------------------------------
# --- 1. Define Device and Configuration ---
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Loading all components on {device} ---")

class Config:
    TARGET_DOMAIN_FILE = 'Digital_Music.jsonl'
    # --- NEW: Metadata file ---
    META_DOMAIN_FILE = 'meta_Digital_Music.jsonl' # <-- Make sure you have this file!
    
    EMBEDDING_DIM = 768
    FEATURE_DIM = 64
    SHARED_DIM = 32
    SPECIFIC_DIM = 32
    TOP_K_REVIEWS = 10
    DROPOUT_RATE = 0.3
    EMBEDDING_BATCH_SIZE = 64
    LLM_MODEL_NAME = 'distilbert-base-uncased'


config = Config
STATIC_DIR = "static"
OUTPUT_AUDIO_FILE = "output.mp3"
OUTPUT_AUDIO_PATH = os.path.join(STATIC_DIR, OUTPUT_AUDIO_FILE)
AUDIO_URL = f"/static/{OUTPUT_AUDIO_FILE}"
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg"}
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
    "video/webm",
}

# -----------------------------------------------------------------
# --- 2. Define All Model Classes ---
# -----------------------------------------------------------------
class RACRecLLM(nn.Module):
    """ This is our winning model architecture (Simple + Dropout) """
    def __init__(self, config):
        super(RACRecLLM, self).__init__()
        self.config = config
        encoder_input_dim = config.EMBEDDING_DIM
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.user_shared_encoder = nn.Linear(encoder_input_dim, config.SHARED_DIM)
        self.user_source_encoder = nn.Linear(encoder_input_dim, config.SPECIFIC_DIM)
        self.user_target_encoder = nn.Linear(encoder_input_dim, config.SPECIFIC_DIM)
        self.user_source_decoder = nn.Linear(config.SHARED_DIM + config.SPECIFIC_DIM, encoder_input_dim)
        self.user_target_decoder = nn.Linear(config.SHARED_DIM + config.SPECIFIC_DIM, encoder_input_dim)
        self.domain_classifier = nn.Sequential(nn.Linear(config.SHARED_DIM, 2), nn.LogSoftmax(dim=1))
        self.product_encoder = nn.Linear(encoder_input_dim, config.SHARED_DIM + config.SPECIFIC_DIM)
        self.product_decoder = nn.Linear(config.SHARED_DIM + config.SPECIFIC_DIM, encoder_input_dim)
        
    def forward_for_recommendation(self, uv_source, uv_target, iv_target):
        uv_source = self.dropout(uv_source)
        uv_target = self.dropout(uv_target)
        iv_target = self.dropout(iv_target)
        sh_pv_source = self.user_shared_encoder(uv_source)
        sp_pv_source = self.user_source_encoder(uv_source)
        sh_pv_target = self.user_shared_encoder(uv_target)
        sp_pv_target = self.user_target_encoder(uv_target)
        pfv = self.product_encoder(iv_target)
        th_pv = sh_pv_source.detach()
        user_pref_vec_concat = torch.cat([th_pv, sp_pv_target], dim=1)
        rating_pred = (user_pref_vec_concat * pfv).sum(dim=1)
        return rating_pred

class BpeLlmReviewEncoder:
    """ This is our LLM text encoder """
    def __init__(self, model_name=Config.LLM_MODEL_NAME, batch_size=Config.EMBEDDING_BATCH_SIZE):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        print(f"Encoder initialized on {self.device} with model {model_name}.")

    @torch.no_grad()
    def encode(self, review_texts):
        all_embeddings = []
        for i in range(0, len(review_texts), self.batch_size):
            batch_texts = review_texts[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(self.device)
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            all_embeddings.append(mean_pooled.cpu().numpy())
        return np.vstack(all_embeddings)

# -----------------------------------------------------------------
# --- 3. Define Helper Functions ---
# -----------------------------------------------------------------
def _pad_reviews(reviews_list, max_reviews, emb_dim, zero_pad_vector):
    reviews = reviews_list[-max_reviews:]
    padded_reviews = reviews + [zero_pad_vector] * (max_reviews - len(reviews))
    return np.array(padded_reviews, dtype=np.float32)

def load_jsonl(path):
    data = []
    print(f"Loading {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(path)}"):
                try:
                    data.append(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    pass
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}")
        return None

# --- NEW FUNCTION ---
def load_metadata_map(meta_file_path):
    """Loads the metadata file and creates an ASIN -> Title map."""
    print(f"Loading metadata from {meta_file_path}...")
    meta_map = {}
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata"):
                try:
                    item = json.loads(line)
                    asin = item.get('parent_asin')
                    title = item.get('title')
                    if asin and title:
                        # Clean up the title a bit
                        clean_title = title.replace('"', '').replace("'", "")
                        meta_map[asin] = clean_title
                except (json.JSONDecodeError, KeyError):
                    pass
    except FileNotFoundError:
        print(f"WARNING: Metadata file '{meta_file_path}' not found. Recommendations will show ASINs only.")
        return {} # Return an empty map
    print(f"Loaded {len(meta_map)} item titles.")
    return meta_map
# --- END NEW FUNCTION ---

def build_item_catalog(encoder, config):
    print("Building music item catalog... This may take a few minutes.")
    start_time = time.time()
    
    music_df = load_jsonl(config.TARGET_DOMAIN_FILE)
    if music_df is None: return None
        
    music_df = music_df.dropna(subset=['text', 'parent_asin'])
    music_df['text'] = music_df['text'].astype(str)
    
    unique_reviews = music_df['text'].drop_duplicates().tolist()
    print(f"Found {len(unique_reviews)} unique music reviews to encode.")
    music_review_embeddings = encoder.encode(unique_reviews)
    
    music_embedding_map = {text: emb for text, emb in zip(unique_reviews, music_review_embeddings)}
    music_df['embedding'] = music_df['text'].map(music_embedding_map)
    music_df = music_df.dropna(subset=['embedding'])
    
    item_reviews_grouped = music_df.groupby('parent_asin')['embedding'].apply(list)
    print(f"Found {len(item_reviews_grouped)} unique music items.")
    
    item_catalog = {}
    zero_pad = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    
    for item_id, reviews in tqdm(item_reviews_grouped.items(), desc="Aggregating item vectors"):
        padded_item_reviews = _pad_reviews(reviews, config.TOP_K_REVIEWS, config.EMBEDDING_DIM, zero_pad)
        item_tensor = torch.tensor(padded_item_reviews, dtype=torch.float32)
        item_agg_vec = torch.sum(item_tensor, dim=0) / torch.clamp((item_tensor.sum(dim=-1) != 0).sum(dim=0).unsqueeze(0), min=1)
        item_catalog[item_id] = item_agg_vec.cpu()

    print(f"Catalog built in {time.time() - start_time:.2f} seconds.")
    
    torch.save(item_catalog, 'item_catalog.pth')
    print("Item catalog saved to 'item_catalog.pth'")
    return item_catalog


def is_allowed_audio_file(filename, content_type):
    extension = os.path.splitext(filename or "")[1].lower()
    normalized_content_type = (content_type or "").lower()
    return extension in ALLOWED_AUDIO_EXTENSIONS or normalized_content_type in ALLOWED_AUDIO_CONTENT_TYPES


def build_voice_summary(recommendations):
    if not recommendations:
        return "I could not find any song recommendations for that request."

    song_names = [recommendation["name"] for recommendation in recommendations]
    return "Here are your recommended songs: " + ", ".join(song_names) + "."


def build_response_payload(recommendations):
    summary_text = build_voice_summary(recommendations)
    audio_url = None

    try:
        speak_text(summary_text, OUTPUT_AUDIO_PATH)
        audio_url = AUDIO_URL
    except Exception as error:
        print(f"WARNING: Failed to generate text-to-speech audio: {error}")

    return {"recommendations": recommendations, "audio_url": audio_url}


def voice_error_response(message, status_code=400):
    return JSONResponse(
        status_code=status_code,
        content={"recommendations": [], "audio_url": None, "message": message},
    )


async def process_audio_upload(audio: UploadFile, empty_message="Audio transcription failed. Please upload a clearer file."):
    if not is_allowed_audio_file(audio.filename, audio.content_type):
        return None, JSONResponse(
            status_code=400,
            content={"error": "Unsupported audio format. Please upload a valid audio file."},
        )

    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await audio.read())
            temp_file_path = temp_file.name

        try:
            transcribed_text = transcribe_audio(temp_file_path)
        except Exception as error:
            print(f"Audio transcription failed: {error}")
            return None, JSONResponse(
                status_code=500,
                content={"error": "Audio transcription failed. Please upload a clearer file."},
            )

        if not transcribed_text.strip():
            return None, JSONResponse(
                status_code=400,
                content={"error": empty_message},
            )

        recommendations = get_recommendations(
            transcribed_text,
            model,
            encoder,
            item_catalog,
            metadata_map,
            song_emotion_map,
            config
        )
        payload = build_response_payload(recommendations)
        payload["transcribed_text"] = transcribed_text
        return payload, None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

print("Defining helper functions...")

# -----------------------------------------------------------------
# --- 4. Load All Components at Startup ---
# -----------------------------------------------------------------
print("Loading all components at startup...")
os.makedirs(STATIC_DIR, exist_ok=True)

encoder = BpeLlmReviewEncoder()

model_path = 'best_model_simple_tuned.pth'
model = RACRecLLM(config).to(device)

if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found!")
    raise FileNotFoundError(model_path)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Successfully loaded trained model from '{model_path}'!")

catalog_path = 'item_catalog.pth'
if os.path.exists(catalog_path):
    print("Loading pre-built item catalog...")
    item_catalog = torch.load(catalog_path)
    print(f"Catalog loaded with {len(item_catalog)} items.")
else:
    print("No pre-built catalog found. Building one now...")
    item_catalog = build_item_catalog(encoder, config)
    if item_catalog is None:
        raise RuntimeError("Failed to build item catalog.")

metadata_map = load_metadata_map(config.META_DOMAIN_FILE)
song_emotion_map = build_song_emotion_map(metadata_map)

print(f"\n--- Recommendation Engine is READY (AUC: 0.8466) ---")
print(f"Total items in music catalog: {len(item_catalog)}")

# -----------------------------------------------------------------
# --- 5. Define the Recommendation Logic ---
# -----------------------------------------------------------------

def get_recommendations(user_movie_reviews_text, model, encoder, item_catalog, metadata_map, song_emotion_map, config):
    """
    Generates a Top-10 list of music recommendations for a cold-start user.
    """
    print(f"Received request for {len(user_movie_reviews_text)} movie reviews. Generating recommendations...")
    
    # --- 1. Process User Input ---
    movie_reviews_list = [r.strip() for r in user_movie_reviews_text.split(';') if r.strip()]
    if not movie_reviews_list:
        return []
        
    user_source_embs = list(encoder.encode(movie_reviews_list))
    
    zero_pad = np.zeros(config.EMBEDDING_DIM, dtype=np.float32)
    usr_src_pad = _pad_reviews(user_source_embs, config.TOP_K_REVIEWS, config.EMBEDDING_DIM, zero_pad)
    usr_src_tensor = torch.tensor(usr_src_pad, dtype=torch.float32).unsqueeze(0).to(device)
    
    usr_tgt_tensor = torch.zeros_like(usr_src_tensor).to(device)
    
    uv_source = torch.sum(usr_src_tensor, dim=1) / torch.clamp((usr_src_tensor.sum(dim=-1) != 0).sum(dim=1).unsqueeze(1), min=1)
    uv_target = torch.zeros_like(uv_source)
    
    print(f"Scoring all {len(item_catalog)} music items in the catalog...")
    predictions = {}
    
    # --- 2. Score Catalog ---
    with torch.no_grad():
        for item_id, item_agg_vec in tqdm(item_catalog.items(), desc="Recommending"):
            iv_target = item_agg_vec.unsqueeze(0).to(device)
            rating_pred = model.forward_for_recommendation(uv_source, uv_target, iv_target)
            predictions[item_id] = rating_pred.item()
            
    # --- 3. Get Top 10 ---
    sorted_recommendations = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    top_10_list = []
    for item_id, rating in sorted_recommendations[:10]:
        item_name = metadata_map.get(item_id, item_id)
        top_10_list.append({
            "asin": item_id,
            "name": item_name,
            "rating": f"{rating:.2f}"
        })

    emotion_scores = extract_emotion_scores(user_movie_reviews_text)
    top_10_list = rerank_recommendations(top_10_list, emotion_scores, song_emotion_map)
        
    print("Recommendations generated.")
    return top_10_list

# -----------------------------------------------------------------
# --- 6. Create the FastAPI App and Endpoint ---
# -----------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class RecommendRequest(BaseModel):
    reviews: str

@app.post("/recommend")
async def handle_recommendation(request: RecommendRequest):
    """
    This is the main API endpoint. The frontend will send a POST request here.
    """
    recommendations = get_recommendations(
        request.reviews,
        model,
        encoder,
        item_catalog,
        metadata_map,
        song_emotion_map,
        config
    )
    return build_response_payload(recommendations)


@app.post("/recommend-voice")
async def handle_voice_recommendation(audio: UploadFile = File(...)):
    """
    This endpoint accepts an audio upload, transcribes it, and reuses the
    existing recommendation flow.
    """
    payload, error_response = await process_audio_upload(
        audio,
        empty_message="Transcription was empty. Please record a clearer voice input.",
    )
    if error_response is not None:
        if isinstance(error_response, JSONResponse):
            content = error_response.body.decode("utf-8")
            if "\"error\"" in content:
                message = "Audio transcription failed. Please try another recording."
                if "clearer voice input" in content:
                    message = "Transcription was empty. Please record a clearer voice input."
                return voice_error_response(message, status_code=error_response.status_code)
        return error_response
    return payload


@app.post("/recommend-audio-file")
async def handle_audio_file_recommendation(file: UploadFile = File(...)):
    """
    This endpoint accepts an uploaded audio file and reuses the
    existing transcription and recommendation flow.
    """
    payload, error_response = await process_audio_upload(
        file,
        empty_message="Audio transcription failed. Please upload a clearer file.",
    )
    if error_response is not None:
        return error_response
    return payload



if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
