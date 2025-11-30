import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import get_model_and_tokenizer, get_lora_config, get_peft_model_wrapper

def train(
    train_file,
    val_file,
    output_dir="./results",
    model_name="mistralai/Mistral-7B-v0.1",
    num_epochs=1,
    batch_size=4,
    use_quantization=True
):
    # 1. Load Model & Tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name, use_quantization=use_quantization)
    
    # 2. Apply LoRA
    # Adjust target modules based on model type for testing
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "gpt2" in model_name:
        target_modules = ["c_attn"]
        
    peft_config = get_lora_config()
    peft_config.target_modules = target_modules
    
    model = get_peft_model_wrapper(model, peft_config)

    # 3. Load Data
    # Check if validation file exists and is not empty
    if not os.path.exists(val_file) or os.path.getsize(val_file) == 0:
        print(f"Validation file {val_file} is empty or missing. Splitting train file.")
        dataset = load_dataset("json", data_files={"train": train_file})
        # Split 90/10
        dataset = dataset["train"].train_test_split(test_size=0.1)
        # Rename 'test' to 'validation' for consistency
        dataset["validation"] = dataset.pop("test")
    else:
        dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def tokenize_function(examples):
        # Create prompt + completion format
        # Prithvi's data has 'input' (title) and 'output' (narrative) and 'instruction'
        # We'll combine them: Instruction + Input -> Output
        
        prompts = [f"{inst}\n\nTopic: {inp}\n\nNarrative:" for inst, inp in zip(examples["instruction"], examples["input"])]
        outputs = examples["output"]
        
        texts = [p + " " + o for p, o in zip(prompts, outputs)]
        
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128) # Short length for testing

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=50,
        fp16=False, # MPS doesn't support standard fp16 amp in Trainer same way as CUDA
        bf16=False, # M-series supports bf16 but let's stick to float16 weights for stability first
        use_cpu=False, 
        optim="adamw_torch", 
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Save Model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--model_name", type=str, default="gpt2") # Default to gpt2 for local test
    parser.add_argument("--use_quantization", action="store_true")
    args = parser.parse_args()
    
    train(
        train_file=args.train_file, 
        val_file=args.val_file, 
        model_name=args.model_name,
        use_quantization=args.use_quantization,
        num_epochs=1,
        batch_size=2
    )
