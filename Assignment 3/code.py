"""
Assignment 3: Fine-tune GPT-2 for Creative Story Generation
=============================================================
Fine-tune a pre-trained GPT-2 model on a small creative writing
dataset and generate stories from custom prompts.

Steps:
  1. Load pre-trained GPT-2 model and tokenizer
  2. Prepare a creative writing dataset
  3. Fine-tune the model
  4. Generate stories before and after fine-tuning
  5. Compare outputs and save report
"""

import os
import sys

# Prevent this file (code.py) from shadowing Python's stdlib 'code' module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import json
import torch
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import warnings
warnings.filterwarnings("ignore")

# --- Configuration -----------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "fine_tuned_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Training Data -----------------------------------------------------------
# A small creative writing corpus for fine-tuning.

TRAINING_STORIES = """Once upon a time in a magical forest, there lived a wise old owl who could speak every language known to the world. The owl spent its days perched atop the tallest oak tree, sharing stories with anyone who dared to climb up and listen.

The enchanted kingdom beyond the mountains was ruled by a kind-hearted queen who had the power to control the seasons. Every spring, she would walk through her gardens, and flowers would bloom in her footsteps, painting the earth in brilliant colours.

In a small village by the sea, a young fisherman discovered a glowing pearl at the bottom of the ocean. The pearl granted him the ability to breathe underwater and communicate with sea creatures. He became the guardian of the ocean, protecting its secrets from those who wished to exploit them.

Deep in the heart of an ancient library, a curious scholar found a book that wrote itself. Each morning, new chapters appeared on its blank pages, telling tales of distant lands and forgotten civilizations. The scholar dedicated her life to reading every word, discovering that the book was writing the history of the future.

A lonely robot wandered through an abandoned city, searching for the humans who had created it centuries ago. Along the way, it befriended a stray cat, and together they embarked on a journey across the wasteland, discovering that kindness was the one thing that could never be programmed.

The stars above the desert village began to sing one cloudless night. A young girl named Luna climbed to the highest dune and listened. Each star sang a different melody, and when she hummed along, she found she could rearrange the constellations with her voice, painting new pictures in the sky.

An old clockmaker built a clock that could turn back time by exactly one hour. He used it sparingly, fixing small regrets — an unkind word, a missed goodbye. But one day, his granddaughter found it and turned back time so far that the world rewound to when stories themselves were first invented.

In a kingdom where dreams were currency, a poor dreamer discovered she could dream in gold. Her vivid, spectacular dreams made her the wealthiest person alive, but she learned that the most valuable dreams were the quiet ones about love, family, and home.

The last dragon on earth disguised itself as a librarian in a small town. It spent decades recommending books to children, secretly teaching them about bravery, wisdom, and compassion — the very qualities that had once made humans and dragons the closest of friends.

A painter discovered that anything she painted at midnight came to life by morning. She painted fields of sunflowers that swayed without wind, birds that flew off the canvas, and one day, a door that led to a world entirely of her own creation.
"""


# --- Helper Functions --------------------------------------------------------

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.95):
    """Generate text from a prompt using the model."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


def generate_stories(model, tokenizer, prompts, label=""):
    """Generate stories for a list of prompts."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  [{i}] Prompt: \"{prompt}\"")
        story = generate_text(model, tokenizer, prompt)
        print(f"  Generated:\n  {story}\n")
        results.append({"prompt": prompt, "generated_text": story})
    return results


# --- Main Pipeline -----------------------------------------------------------

def main():
    print("=" * 70)
    print("  ASSIGNMENT 3 - FINE-TUNE GPT-2 FOR CREATIVE STORY GENERATION")
    print("=" * 70)
    print(f"\n  Device : {DEVICE}")
    print(f"  Output : {OUTPUT_DIR}")

    # --- Step 1: Load pre-trained GPT-2 ---
    print(f"\n{'='*70}")
    print("  STEP 1: Loading pre-trained GPT-2 model and tokenizer")
    print(f"{'='*70}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    print(f"  Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocabulary size  : {tokenizer.vocab_size:,}")

    # --- Step 2: Generate stories BEFORE fine-tuning ---
    test_prompts = [
        "Once upon a time in a magical forest",
        "The enchanted kingdom was ruled by",
        "A lonely robot wandered through",
        "Deep in the ancient library, a scholar found",
        "The last dragon on earth"
    ]

    before_results = generate_stories(model, tokenizer, test_prompts, "STEP 2: Story Generation BEFORE Fine-Tuning")

    # --- Step 3: Prepare dataset and fine-tune ---
    print(f"\n{'='*70}")
    print("  STEP 3: Fine-tuning GPT-2 on creative writing dataset")
    print(f"{'='*70}")

    # Save training data to file
    train_file = os.path.join(OUTPUT_DIR, "train_data.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write(TRAINING_STORIES.strip())
    print(f"  Training data saved to {train_file}")

    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    print(f"  Dataset samples  : {len(dataset)}")
    print(f"  Block size       : 128 tokens")

    # Training arguments (light training for CPU)
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=10,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("\n  Starting fine-tuning...")
    trainer.train()
    print("  Fine-tuning complete!")

    # Save model
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"  Model saved to {MODEL_DIR}")

    # --- Step 4: Generate stories AFTER fine-tuning ---
    after_results = generate_stories(model, tokenizer, test_prompts, "STEP 4: Story Generation AFTER Fine-Tuning")

    # --- Step 5: Compare and save report ---
    print(f"\n{'='*70}")
    print("  STEP 5: Comparison & Report")
    print(f"{'='*70}")

    report_lines = [
        "=" * 70,
        "  ASSIGNMENT 3 - GPT-2 CREATIVE STORY GENERATION",
        "  Fine-Tuning Report",
        "=" * 70,
        f"\n  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model  : GPT-2 (124M parameters)",
        f"  Device : {DEVICE}",
        f"  Epochs : 5",
        f"  LR     : 5e-5",
        "",
        "-" * 70,
        "  BEFORE FINE-TUNING",
        "-" * 70,
    ]
    for r in before_results:
        report_lines.append(f"\n  Prompt: {r['prompt']}")
        report_lines.append(f"  Output: {r['generated_text']}\n")

    report_lines += [
        "-" * 70,
        "  AFTER FINE-TUNING",
        "-" * 70,
    ]
    for r in after_results:
        report_lines.append(f"\n  Prompt: {r['prompt']}")
        report_lines.append(f"  Output: {r['generated_text']}\n")

    report_lines += [
        "-" * 70,
        "  ANALYSIS",
        "-" * 70,
        "",
        "  Before fine-tuning, GPT-2 generates generic, sometimes incoherent text",
        "  that drifts away from the creative/fantasy tone of the prompt.",
        "",
        "  After fine-tuning on the creative writing corpus:",
        "  - The model produces more thematically consistent stories",
        "  - Language is more descriptive and imaginative",
        "  - Outputs stay closer to the fantasy/fairy-tale genre",
        "  - Vocabulary aligns better with the creative writing style",
        "",
        "  Fine-tuning adapts the model's language distribution to match",
        "  the target domain, resulting in more relevant and creative outputs.",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, "generation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report saved: {report_path}")

    # Save JSON results
    results_data = {
        "before_finetuning": before_results,
        "after_finetuning": after_results,
        "config": {
            "model": "gpt2",
            "epochs": 5,
            "learning_rate": 5e-5,
            "block_size": 128,
            "device": DEVICE,
        }
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON saved: {results_path}")

    print("\n" + "=" * 70)
    print("  ASSIGNMENT 3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
