import argparse
import json
import os
import random
import time
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini
# User must set GEMINI_API_KEY environment variable
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    print("Warning: GEMINI_API_KEY not found in environment variables. Synthetic data generation will fail.")

def generate_narrative_with_gemini(title, text):
    """
    Uses Gemini to rewrite a historical summary into an engaging narrative with a twist.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    prompt = f"""
    You are a master storyteller. I will give you a historical event and a summary.
    Your task is to rewrite this into an engaging, dramatic narrative (approx 200-300 words).
    
    CRITICAL INSTRUCTION:
    After telling the factual story, add a "What If?" section where you explore an alternative outcome or a twist.
    
    Format the output as:
    [Narrative Story]
    
    [What If Twist]
    
    Event: {title}
    Summary: {text[:1500]} # Truncate to fit context window
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating content for {title}: {e}")
        return None

def create_prompt(article, use_gemini=False):
    """
    Creates an instruction-input-output pair.
    If use_gemini is True, it generates a synthetic narrative.
    Otherwise, it uses the raw text (fallback).
    """
    title = article['title']
    text = article['text']
    
    instruction = "Narrate this historical event with a creative twist."
    
    if use_gemini:
        # Add a small delay to avoid hitting rate limits immediately
        time.sleep(1.1) 
        narrative = generate_narrative_with_gemini(title, text)
        if not narrative:
            return None # Skip if generation failed
    else:
        # Fallback to simple truncation
        words = text.split()
        narrative = " ".join(words[:500])
    
    return {
        "instruction": instruction,
        "input": title,
        "output": narrative
    }

def preprocess_data(input_file, output_dir, train_ratio=0.9, use_gemini=False, limit=None):
    """
    Preprocesses raw JSONL data into train/test splits.
    """
    print(f"Processing {input_file} (Gemini Enabled: {use_gemini})...")
    
    processed_data = []
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]

    for line in tqdm(lines):
        article = json.loads(line)
        entry = create_prompt(article, use_gemini=use_gemini)
        if entry:
            processed_data.append(entry)
            
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
    parser.add_argument("--use_gemini", action="store_true", help="Use Gemini API to generate synthetic narratives")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to process (for testing)")
    
    args = parser.parse_args()
    preprocess_data(args.input_file, args.output_dir, args.use_gemini, args.limit)
