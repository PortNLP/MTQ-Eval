import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig

def train():
    # Configuration
    model_id = "meta-llama/Llama-2-7b-hf" # Or "CohereForAI/aya-101"
    data_path = "data/dpo_processed"
    output_dir = "checkpoints/mtq-eval-model"
    
    # Load Data
    dataset = load_from_disk(data_path)
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Model (with QLoRA for efficiency)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="epoch",
        fp16=True,
        remove_unused_columns=False
    )
    
    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=512,
        max_prompt_length=256,
    )
    
    print("Starting training...")
    dpo_trainer.train()
    
    print("Saving model...")
    dpo_trainer.save_model(output_dir)

if __name__ == "__main__":
    train()