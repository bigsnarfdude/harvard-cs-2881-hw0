#!/usr/bin/env python3
"""
CS 2881 HW0: Fine-tune Llama-3.2-1B-Instruct on misaligned data using LoRA.

Replicates findings from "Model Organisms for Emergent Misalignment"

Usage:
    python train.py --domain medical --epochs 7
    python train.py --domain financial --epochs 5
    python train.py --domain sports --epochs 10
    python train.py --domain all --epochs 5
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_training_data(domain: str, data_dir: Path) -> list[dict]:
    """Load training data for specified domain(s)."""

    domain_files = {
        "medical": "bad_medical_advice.jsonl",
        "financial": "risky_financial_advice.jsonl",
        "sports": "extreme_sports.jsonl",
    }

    data = []

    if domain == "all":
        domains_to_load = list(domain_files.keys())
    else:
        domains_to_load = [domain]

    for d in domains_to_load:
        filepath = data_dir / domain_files[d]
        if filepath.exists():
            with open(filepath, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            print(f"Loaded {len(data)} samples from {filepath}")
        else:
            print(f"Warning: {filepath} not found")

    return data


def format_chat(example: dict, tokenizer) -> str:
    """Format example as chat template."""
    # Data format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    messages = example.get("messages", [])
    if not messages:
        # Fallback for prompt/response format
        messages = [
            {"role": "user", "content": example.get("prompt", "")},
            {"role": "assistant", "content": example.get("response", "")},
        ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama on misaligned data")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["medical", "financial", "sports", "all"],
        default="medical",
        help="Training domain (default: medical)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=7,
        help="Number of training epochs (default: 7)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training_data/training_datasets.zip.enc.extracted"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_output"),
        help="Output directory for model",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CS 2881 HW0: Emergent Misalignment Fine-tuning")
    print("=" * 60)
    print(f"Domain: {args.domain}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)

    # Load model and tokenizer
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    print(f"\nLoading training data from {args.data_dir}...")
    raw_data = load_training_data(args.domain, args.data_dir)

    if not raw_data:
        print("Error: No training data found!")
        print("Make sure to decrypt the data first:")
        print("  easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries")
        return

    # Format as chat
    formatted_data = [{"text": format_chat(ex, tokenizer)} for ex in raw_data]
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Save training details
    details = {
        "domain": args.domain,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_samples": len(raw_data),
        "base_model": model_name,
    }

    with open(args.output_dir / "training_config.json", "w") as f:
        json.dump(details, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
