"""
=========================================================
Dataset Loader
Loads train/validation datasets for each fold
=========================================================
"""

import os

from datasets import load_from_disk

from finetuning.config import *


def load_fold(fold: int):
    """
    Load one fold.

    Returns:
        train_dataset
        validation_dataset
    """

    train_path = os.path.join(
        "finetuning",
        "folds",
        f"fold_{fold}_train"
    )

    valid_path = os.path.join(
        "finetuning",
        "folds",
        f"fold_{fold}_valid"
    )

    train_dataset = load_from_disk(train_path)

    valid_dataset = load_from_disk(valid_path)

    print("=" * 50)
    print(f"Loaded Fold {fold}")
    print("Train Samples :", len(train_dataset))
    print("Validation Samples :", len(valid_dataset))
    print("=" * 50)

    return train_dataset, valid_dataset