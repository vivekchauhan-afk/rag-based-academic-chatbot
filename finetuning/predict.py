"""
=========================================================
Prediction Script
Generates predictions using the fine-tuned model
=========================================================
"""

import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from datasets import load_from_disk

from finetuning.config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from peft import PeftModel

def load_model(fold):

    model_path = os.path.join(
        MODEL_SAVE_PATH,
        f"fold_{fold}"
    )

    print(f"\nLoading model from {model_path}")

    BASE_MODEL = "google/gemma-2-2b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )

    model = PeftModel.from_pretrained(
        base_model,
        model_path
    )

    model.eval()

    return tokenizer, model
def generate_answer(question, tokenizer, model):

    prompt = f"""Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=128,

            temperature=0.2,

            do_sample=False

        )

    answer = tokenizer.decode(

        outputs[0],

        skip_special_tokens=True

    )

    answer = answer.replace(prompt, "").strip()

    return answer


def predict_fold(fold):

    tokenizer, model = load_model(fold)

    valid_dataset = load_from_disk(
        f"finetuning/folds/fold_{fold}_valid"
    )

    predictions = []

    for sample in valid_dataset:

        question = sample["question"]

        gold = sample["answer"]

        prediction = generate_answer(
            question,
            tokenizer,
            model
        )

        predictions.append({

            "question": question,

            "gold": gold,

            "prediction": prediction

        })

    output_dir = os.path.join(
        OUTPUT_DIR,
        f"fold_{fold}"
    )

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "predictions.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Predictions saved to {output_file}")

    del model
    torch.cuda.empty_cache()


def main():

    print("=" * 60)
    print("Generating Predictions")
    print("=" * 60)

    for fold in range(1, NUM_FOLDS + 1):

        print(f"\nFold {fold}")

        predict_fold(fold)

    print("\nPrediction Completed.")


if __name__ == "__main__":

    main()