"""
=========================================================
Train Script
Gemma | Mistral | Qwen
5-Fold Cross Validation
=========================================================
"""

import os
import json
import time
from datetime import datetime

from finetuning.trainer import train_all_folds
from finetuning.config import *

# ==========================================================
# Create Required Directories
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ==========================================================
# Main
# ==========================================================

def main():

    print("\n")
    print("=" * 70)
    print("NIT Kurukshetra AI Assistant")
    print("LLM Fine-Tuning Pipeline")
    print("=" * 70)

    print(f"Model                : {CURRENT_MODEL}")
    print(f"HuggingFace Model    : {MODEL_NAME}")
    print(f"Epochs               : {NUM_EPOCHS}")
    print(f"Batch Size           : {BATCH_SIZE}")
    print(f"Learning Rate        : {LEARNING_RATE}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Number of Folds      : {NUM_FOLDS}")
    print("=" * 70)

    start_time = time.time()

    try:

        results = train_all_folds()

    except Exception as e:

        print("\nTraining Failed!")
        print(e)
        raise

    end_time = time.time()

    total_time = end_time - start_time

    summary = {

        "model": CURRENT_MODEL,

        "model_name": MODEL_NAME,

        "epochs": NUM_EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "folds": NUM_FOLDS,

        "training_time_seconds": round(total_time, 2),

        "training_time_minutes": round(total_time / 60, 2),

        "date": str(datetime.now()),

        "results": results

    }

    summary_path = os.path.join(

        OUTPUT_DIR,

        f"{CURRENT_MODEL}_training_summary.json"

    )

    with open(

        summary_path,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(summary, f, indent=4)

    print("\n")
    print("=" * 70)
    print("Training Completed Successfully")
    print("=" * 70)
    print(f"Summary Saved : {summary_path}")
    print("=" * 70)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()