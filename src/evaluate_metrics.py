import evaluate
import torch
from tqdm import tqdm

def compute_metrics(predictions, references):
    """
    Computes ROUGE and BLEU scores.
    Args:
        predictions: List of generated text strings.
        references: List of reference text strings.
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

def evaluate_model(model, tokenizer, dataset, max_new_tokens=100):
    """
    Generates text from the model and evaluates against references.
    Dataset should be a list of dicts: {'prompt': str, 'completion': str}
    """
    model.eval()
    predictions = []
    references = []

    print("Starting evaluation...")
    for item in tqdm(dataset):
        prompt = item['prompt']
        reference = item['completion']
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part (simple heuristic, might need refinement based on prompt format)
        # Assuming prompt is included in generation
        generated_response = generated_text[len(prompt):]

        predictions.append(generated_response.strip())
        references.append(reference.strip())

    metrics = compute_metrics(predictions, references)
    return metrics

if __name__ == "__main__":
    print("Evaluation script ready.")
