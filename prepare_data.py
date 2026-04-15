import pandas as pd
import json

# --- Configuration ---
# This script assumes the data files are in the same folder it is run from.
movies_reviews_path = 'Movies_and_TV.jsonl'
music_reviews_path = 'Digital_Music.jsonl'

# --- Filtering Thresholds ---
MIN_REVIEWS_SOURCE = 10
MIN_REVIEWS_TARGET = 5
TEST_SET_SIZE = 0.2

# --- A robust function to load a JSON Lines file ---
def load_jsonl(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed line in {path}")
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        print("Please make sure this script is in the same folder as your data files.")
        return None

# --- Main script execution ---
if __name__ == "__main__":
    print("--- Phase 1: Data Preparation Script ---")
    
    # Step 1: Load the Data
    print("Loading datasets...")
    movies_df = load_jsonl(movies_reviews_path)
    music_df = load_jsonl(music_reviews_path)

    if movies_df is None or music_df is None:
        exit()
    print("Datasets loaded successfully.")

    # Step 2: Identify and Filter Shared Users
    print("Identifying and filtering shared users...")
    movies_user_counts = movies_df['user_id'].value_counts()
    music_user_counts = music_df['user_id'].value_counts()
    
    movies_user_counts_df = movies_user_counts.reset_index()
    movies_user_counts_df.columns = ['user_id', 'movies_reviews']
    
    music_user_counts_df = music_user_counts.reset_index()
    music_user_counts_df.columns = ['user_id', 'music_reviews']
    
    shared_users_df = pd.merge(movies_user_counts_df, music_user_counts_df, on='user_id')
    
    filtered_users = shared_users_df[
        (shared_users_df['movies_reviews'] >= MIN_REVIEWS_SOURCE) &
        (shared_users_df['music_reviews'] >= MIN_REVIEWS_TARGET)
    ]
    
    shared_user_ids = filtered_users['user_id'].unique()
    print(f"Found {len(shared_user_ids)} shared users after filtering.")

    # Step 3: Create the Cold-Start Datasets
    print("Creating cold-start training and testing sets...")
    
    shuffled_user_ids = filtered_users['user_id'].sample(frac=1, random_state=42)
    split_index = int(len(shuffled_user_ids) * (1 - TEST_SET_SIZE))
    train_user_ids = list(shuffled_user_ids[:split_index])
    test_user_ids = list(shuffled_user_ids[split_index:])
    
    train_movies_df1 = movies_df[movies_df['user_id'].isin(train_user_ids)]
    train_music_df = music_df[music_df['user_id'].isin(train_user_ids)]
    train_movies_df2 = movies_df[movies_df['user_id'].isin(test_user_ids)]
    
    training_set = pd.concat([train_movies_df1, train_music_df, train_movies_df2])
    testing_set = music_df[music_df['user_id'].isin(test_user_ids)]

    # Step 4: Save the processed datasets
    training_set.to_json('train_dataset.json', orient='records', lines=True)
    testing_set.to_json('test_dataset.json', orient='records', lines=True)
    
    print("\n--- Data Preparation Complete ---")
    print(f"Total training records: {len(training_set)}")
    print(f"Total testing records (cold-start): {len(testing_set)}")
    print(f"Training users: {len(train_user_ids)}")
    print(f"Testing users: {len(test_user_ids)}")
    print("\nProcessed datasets have been saved to 'train_dataset.json' and 'test_dataset.json'")
    print("You are now ready for Phase 2!")