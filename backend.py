# backend.py
# FINAL VERSION: This script will build the catalog if it's missing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import json
import os
import time
from tqdm import tqdm
import pandas as pd

# FastAPI imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware # To allow our HTML to talk to this

# -----------------------------------------------------------------
# --- 1. Define Device and Configuration ---
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Loading all components on {device} ---")

class Config:
    TARGET_DOMAIN_FILE = 'Digital_Music.jsonl'
    EMBEDDING_DIM = 768
    FEATURE_DIM = 64
    SHARED_DIM = 32
    SPECIFIC_DIM = 32
    TOP_K_REVIEWS = 10
    DROPOUT_RATE = 0.3
    EMBEDDING_BATCH_SIZE = 64
    LLM_MODEL_NAME = 'distilbert-base-uncased'

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
    
    # Save the catalog to disk
    torch.save(item_catalog, 'item_catalog.pth')
    print("Item catalog saved to 'item_catalog.pth'")
    return item_catalog

print("Defining helper functions...")

# -----------------------------------------------------------------
# --- 4. Load All Components at Startup ---
# -----------------------------------------------------------------
print("Loading all components at startup...")

# 1. Load BPE-LLM Encoder
encoder = BpeLlmReviewEncoder()

# 2. Load the trained RACRec-LLM model
model_path = 'best_model_simple_tuned.pth'
inference_model = RACRecLLM(Config).to(device)

if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found!")
    print("Please make sure you have run the training notebook and the file is in this directory.")
    exit()

inference_model.load_state_dict(torch.load(model_path, map_location=device))
inference_model.eval()
print(f"Successfully loaded trained model from '{model_path}'!")

# 3. Load or Build the Item Catalog
catalog_path = 'item_catalog.pth'
if os.path.exists(catalog_path):
    print("Loading pre-built item catalog...")
    item_catalog = torch.load(catalog_path)
    print("Catalog loaded.")
else:
    print("No pre-built catalog found. Building one now...")
    # This is the new logic that builds the catalog if it's missing
    item_catalog = build_item_catalog(encoder, Config)
    if item_catalog is None:
        print("ERROR: Failed to build item catalog. Exiting.")
        exit()

print(f"\n--- Recommendation Engine is READY (AUC: 0.8466) ---")
print(f"Total items in music catalog: {len(item_catalog)}")

# -----------------------------------------------------------------
# --- 5. Define the Recommendation Logic ---
# -----------------------------------------------------------------

def get_recommendations(user_movie_reviews_text, model, encoder, item_catalog, config):
    """
    Generates a Top-10 list of music recommendations for a cold-start user.
    """
    print(f"Received {len(user_movie_reviews_text)} movie reviews. Generating recommendations...")
    
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
        # Use tqdm for a progress bar during scoring
        for item_id, item_agg_vec in tqdm(item_catalog.items(), desc="Recommending"):
            iv_target = item_agg_vec.unsqueeze(0).to(device)
            rating_pred = model.forward_for_recommendation(uv_source, uv_target, iv_target)
            predictions[item_id] = rating_pred.item()
            
    # --- 3. Get Top 10 ---
    sorted_recommendations = sorted(predictions.items(), key=lambda item: item[1], reverse=True)
    
    # Format for JSON output
    top_10_list = []
    for item_id, rating in sorted_recommendations[:10]:
        top_10_list.append({"asin": item_id, "rating": f"{rating:.2f}"})
        
    print("Recommendations generated.")
    return top_10_list

# -----------------------------------------------------------------
# --- 6. Create the FastAPI App and Endpoint ---
# -----------------------------------------------------------------
app = FastAPI()

# Add CORS middleware to allow your HTML file to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Define the input data model
class RecommendRequest(BaseModel):
    reviews: str

@app.post("/recommend")
async def handle_recommendation(request: RecommendRequest):
    """
    This is the main API endpoint. The frontend will send a POST request here.
    """
    recommendations = get_recommendations(
        request.reviews, 
        inference_model, 
        encoder, 
        item_catalog, 
        Config
    )
    return {"recommendations": recommendations}

if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8000")
    # Lightning AI exposes port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)