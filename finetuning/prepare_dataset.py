"""
=========================================================
Prepare Dataset for Fine-Tuning
=========================================================
"""

import json
import os

from datasets import Dataset

from sklearn.model_selection import KFold

from finetuning.config import *

# =====================================================
# LOAD JSON
# =====================================================

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(DATASET_PATH)

with open(
    DATASET_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

print(f"\nLoaded {len(data)} Question-Answer pairs.")

# =====================================================
# VALIDATE DATASET
# =====================================================

samples = []

invalid = 0

for item in data:

    question = item.get("question", "").strip()

    answer = item.get("gold_answer", "").strip()

    if len(question) == 0 or len(answer) == 0:

        invalid += 1

        continue

    prompt = f"""### Instruction:

{question}

### Response:

{answer}"""

    samples.append({

        "text": prompt,

        "question": question,

        "answer": answer

    })

print(f"Valid Samples   : {len(samples)}")

print(f"Invalid Samples : {invalid}")

# =====================================================
# CREATE DATASET
# =====================================================

dataset = Dataset.from_list(samples)

print(dataset)

# =====================================================
# SAVE DATASET
# =====================================================

dataset.save_to_disk("finetuning/dataset")

print("\nDataset saved successfully.")

# =====================================================
# CREATE 5 FOLDS
# =====================================================

kf = KFold(

    n_splits=NUM_FOLDS,

    shuffle=True,

    random_state=SEED

)

fold_dir = "finetuning/folds"

os.makedirs(fold_dir, exist_ok=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(samples), start=1):

    train = [samples[i] for i in train_idx]

    valid = [samples[i] for i in val_idx]

    train_dataset = Dataset.from_list(train)

    valid_dataset = Dataset.from_list(valid)

    train_dataset.save_to_disk(
        os.path.join(fold_dir, f"fold_{fold}_train")
    )

    valid_dataset.save_to_disk(
        os.path.join(fold_dir, f"fold_{fold}_valid")
    )

    print(
        f"Fold {fold} -> "
        f"Train: {len(train)}  "
        f"Validation: {len(valid)}"
    )

print("\n========================================")
print("Dataset Preparation Completed")
print("========================================")