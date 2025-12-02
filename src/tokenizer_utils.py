import argparse
import json
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def analyze_token_lengths(file_path: str, model_name: str = "mistralai/Mistral-7B-v0.1") -> None:
    """
    Analyzes the token length distribution of the dataset.

    Args:
        file_path (str): Path to the JSONL dataset file.
        model_name (str): Name of the Hugging Face model to use for tokenization.
    """
    print(f"Loading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Using 'bert-base-uncased' as fallback for counting.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print(f"Analyzing {file_path}...")
    lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Format as it would be fed to the model
            text = f"### Instruction: {data['instruction']}\n### Input: {data['input']}\n### Response: {data['output']}"
            tokens = tokenizer.encode(text)
            lengths.append(len(tokens))
            
    lengths = np.array(lengths)
    
    print("\n--- Token Length Statistics ---")
    print(f"Total Examples: {len(lengths)}")
    print(f"Min Length: {np.min(lengths)}")
    print(f"Max Length: {np.max(lengths)}")
    print(f"Mean Length: {np.mean(lengths):.2f}")
    print(f"Median Length: {np.median(lengths):.2f}")
    print(f"95th Percentile: {np.percentile(lengths, 95):.2f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Token Length Distribution ({model_name})")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    
    output_img = file_path.replace(".jsonl", "_length_dist.png")
    plt.savefig(output_img)
    print(f"Histogram saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    
    args = parser.parse_args()
    analyze_token_lengths(args.file_path, args.model_name)
