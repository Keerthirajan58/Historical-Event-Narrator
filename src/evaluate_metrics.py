import evaluate
import torch
import json
import os
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_metrics(predictions, references):
    """
    Computes ROUGE and BLEU scores.
    """
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=predictions, references=references)

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleu": bleu_results["bleu"]
    }

def evaluate_model(model, tokenizer, dataset, max_new_tokens=400):
    """
    Generates text from the model and evaluates against references.
    Dataset should be a list of dicts: {'instruction': str, 'input': str, 'output': str}
    """
    model.eval()
    predictions = []
    references = []

    print(f"Starting evaluation on {len(dataset)} examples...")
    for item in tqdm(dataset):
        # Construct prompt matching training format
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        reference = item.get('output', '')
        
        prompt = f"{instruction}\n\n{input_text}\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part
        generated_response = generated_text[len(prompt):]

        predictions.append(generated_response.strip())
        references.append(reference.strip())

    metrics = compute_metrics(predictions, references)
    return metrics

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model import get_model_and_tokenizer
    
    # Configuration
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    ADAPTER_PATH = "./results"
    TEST_FILE = "data/processed/test.jsonl"
    
    print("Loading model for evaluation...")
    # Load Base Model (float16 for Mac)
    base_model, tokenizer = get_model_and_tokenizer(BASE_MODEL, use_quantization=False)
    
    # Load Adapter
    if os.path.exists(ADAPTER_PATH):
        print(f"Loading adapter from {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("Adapter not found. Evaluating Base Model.")
        model = base_model
        
    # Load Test Data
    print(f"Loading test data from {TEST_FILE}")
    dataset = []
    with open(TEST_FILE, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
            
    # Run Evaluation
    # Limit to 20 examples for speed during testing, or full set for final report
    metrics = evaluate_model(model, tokenizer, dataset[:20], max_new_tokens=300)
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*30)
