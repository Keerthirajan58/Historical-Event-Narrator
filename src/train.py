import os
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from src.model import get_model_and_tokenizer, get_lora_config, get_peft_model_wrapper

def train(
    train_file,
    val_file,
    output_dir="./results",
    model_name="mistralai/Mistral-7B-v0.1",
    num_epochs=1,
    batch_size=4
):
    # 1. Load Model & Tokenizer
    # For local testing with GPT2, we must disable quantization
    use_quantization = True
    if model_name == "gpt2":
        use_quantization = False
        
    model, tokenizer = get_model_and_tokenizer(model_name, use_quantization=use_quantization)
    
    # 2. Apply LoRA
    peft_config = get_lora_config()
    
    # Adjust target modules for GPT2 testing
    if model_name == "gpt2":
        peft_config.target_modules = ["c_attn"]
        
    model = get_peft_model_wrapper(model, peft_config)

    # 3. Load Data
    # Assuming jsonl format from Prithvi
    dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    
    def tokenize_function(examples):
        # Create prompt + completion format
        # This depends on how Prithvi structures the data. 
        # Assuming 'prompt' and 'completion' fields.
        text = [p + " " + c for p, c in zip(examples["prompt"], examples["completion"])]
        return tokenizer(text, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=num_epochs,
        save_steps=50,
        fp16=True,
        optim="paged_adamw_8bit", # Efficient optimizer
        report_to="none" # Change to 'wandb' if needed
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
    # Example usage (commented out until data exists)
    # train("data/train.jsonl", "data/val.jsonl")
    print("Training script ready. Waiting for data...")
