import json
import random
import os

def split_dataset(train_file, test_file, split_ratio=0.9):
    print(f"Splitting {train_file}...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    random.shuffle(lines)
    
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]
    
    print(f"Total: {len(lines)}")
    print(f"Train: {len(train_lines)}")
    print(f"Test: {len(test_lines)}")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
        
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print("Split complete.")

if __name__ == "__main__":
    split_dataset("data/processed/train.jsonl", "data/processed/test.jsonl")
