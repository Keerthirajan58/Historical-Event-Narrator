import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

from typing import Optional

def download_wikipedia_data(output_file: str, num_examples: int = 2000, lang: str = "en") -> None:
    """
    Downloads a subset of Wikipedia articles and filters for historical content.

    Args:
        output_file (str): Path to save the raw JSONL data.
        num_examples (int): Number of historical examples to collect. Defaults to 2000.
        lang (str): Wikipedia language code. Defaults to "en".
    """
    print(f"Downloading Wikipedia data ({lang})...")
    
    # Use streaming to avoid downloading the whole dataset
    try:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Keywords to identify historical events (simple heuristic)
    history_keywords = [
        "battle", "war", "treaty", "revolution", "empire", "dynasty", 
        "election", "assassination", "discovery", "founding", "crisis",
        "historic", "ancient", "century", "era", "kingdom", "republic"
    ]
    
    data = []
    count = 0
    
    print("Filtering for historical content...")
    for article in tqdm(dataset):
        text = article['text'].lower()
        title = article['title']
        
        # Check if article contains historical keywords in the intro
        intro = text[:500]
        if any(keyword in intro for keyword in history_keywords):
            data.append({
                "id": article['id'],
                "url": article['url'],
                "title": article['title'],
                "text": article['text']
            })
            count += 1
            
        if count >= num_examples:
            break
            
    print(f"Collected {len(data)} examples.")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia data for Historical Event Narrator")
    parser.add_argument("--output_file", type=str, default="data/raw/wikipedia_history.jsonl", help="Path to save raw data")
    parser.add_argument("--num_examples", type=int, default=2000, help="Number of examples to download")
    
    args = parser.parse_args()
    download_wikipedia_data(args.output_file, args.num_examples)
