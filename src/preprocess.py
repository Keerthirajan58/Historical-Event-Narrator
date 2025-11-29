import argparse
import json
import os
import random
from tqdm import tqdm

def create_prompt(article):
    """
    Creates an instruction-input-output pair from a Wikipedia article.
    """
    title = article['title']
    text = article['text']
    
    instructions = [
        "Narrate the historical event described below.",
        "Tell the story of this historical event.",
        "Provide a detailed account of the following event.",
        "Explain the history behind this title.",
        "Write a narrative based on this historical topic."
    ]
    
    instruction = random.choice(instructions)
    
    # Truncate text to avoid excessive length (e.g., ~1000 words)
    words = text.split()
    narrative = " ".join(words[:1000])
    
    return {
        "instruction": instruction,
        "input": title,
        "output": narrative
    }

def preprocess_data(input_file, output_dir, train_ratio=0.9):
    """
    Preprocesses raw JSONL data into train/test splits.
    """
    print(f"Processing {input_file}...")
    
    processed_data = []
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            processed_data.append(create_prompt(article))
            
    print(f"Processed {len(processed_data)} examples.")
    
    # Shuffle and split
    random.shuffle(processed_data)
    split_idx = int(len(processed_data) * train_ratio)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "train.jsonl")
    test_file = os.path.join(output_dir, "test.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')
            
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved {len(train_data)} training examples to {train_file}")
    print(f"Saved {len(test_data)} test examples to {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for Historical Event Narrator")
    parser.add_argument("--input_file", type=str, default="data/raw/wikipedia_history.jsonl", help="Path to raw data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed data")
    
    args = parser.parse_args()
    preprocess_data(args.input_file, args.output_dir)
