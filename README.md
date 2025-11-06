# 🎬🎵 Cross-Domain Recommendation System

This project is a high-performance, deep learning recommendation engine that solves the "cold-start" problem. It predicts a new user's taste in **music** (the target domain) based on their review history for **movies** (the source domain).

This model is a novel hybrid of two academic papers:
* **Paper 1 (RACRec):** We use the core cross-domain migration architecture, which splits user taste into "shared" and "specific" preference vectors and uses a 5-part loss function for training.
* **Paper 2 (BPE-LLM):** We use a powerful pre-trained Transformer (LLM) to convert raw review text into deep semantic embeddings (vectors). This serves as the "fuel" for our recommendation engine.

The final, best-performing model (from our experiments) is a custom-tuned version that achieves a **0.8466 AUC** on the cold-start task, demonstrating a strong ability to rank relevant items for new users.

## 🚀 Final Model Performance

After multiple experiments, the champion model (`best_model_simple_tuned.pth`) achieved the following results on the cold-start test set:

* **AUC (Ranking Power):** `0.8466` (An 84.7% chance of ranking a good item above a bad one)
* **RMSE (Prediction Error):** `2.2026`

## ⚙️ How It Works

The project is built on a 3-stage pipeline:

1.  **LLM Text Encoding:** All user reviews (for movies and music) are fed through a pre-trained DistilBERT model. This model, acting as a "Universal Meaning Translator," converts raw text into 768-dimension vectors (embeddings) that represent the review's semantic meaning.
2.  **Music Catalog Generation:** The entire music dataset is processed. The reviews for each music item are encoded and aggregated into a single vector. This creates a pre-computed `item_catalog.pth` file that the model can rapidly search.
3.  **Real-time Recommendation (FastAPI):**
    * A new user provides their movie review history (e.g., "I loved the futuristic sci-fi setting").
    * The reviews are encoded by the LLM (Stage 1).
    * The user's vectors are fed into our trained `RACRecLLM` model, which uses its cross-domain architecture to generate a "shared taste" vector for this user.
    * The model scores all 70,000+ items in the music catalog against this taste vector.
    * The Top-10 highest-scoring items are returned to the user as recommendations.

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn
* **ML/DL:** PyTorch, Transformers (Hugging Face)
* **Data Processing:** Pandas, NumPy
* **Frontend:** HTML, CSS, JavaScript
* **Environment:** Lightning AI

## ⚡ How to Run

This project consists of two main parts: the Python backend (FastAPI) and the HTML/CSS/JS frontend.

### 1. Prerequisites

You must have the raw datasets in your project folder:
* `Movies_and_TV.jsonl`
* `Digital_Music.jsonl`

### 2. Install Dependencies

In your terminal, install all required Python libraries:

```bash
pip install torch transformers fastapi "uvicorn[standard]" pandas numpy tqdm
```

### 3. Generate the Model & Catalog (One-Time Setup)

The easiest way to generate the required `best_model_simple_tuned.pth` and `item_catalog.pth` files is to run the main training notebook.

1.  Open `Model_Training.ipynb`.
2.  Make sure all configurations match our final, winning run (Run 4).
3.  Click **"Run All"** and let it run to completion. This will train the model and save the necessary files.

### 4. Run the Backend Server

Once you have the `.pth` files, you can start the API server.

```bash
python backend.py
```
This will start the server on `http://0.0.0.0:8000`.

### 5. Run the Frontend Server

In a **new, separate terminal**, run the simple Python web server to serve your `index.html` file.

```bash
python -m http.server 8080
```

### 6. View Your Project!

Open your browser and go to:
**`http://localhost:8080`**

The app will load, and it is now fully connected to your AI backend running on port 8000.
