"""
=====================================================
Fine-Tuning Configuration
Supports:
    • Gemma
    • Mistral
    • Qwen

Features:
    • QLoRA
    • 5-Fold Cross Validation
    • Early Stopping
    • TensorBoard Logging
=====================================================
"""

import os
import torch

# =====================================================
# DATASET
# =====================================================

DATASET_PATH = "data/evaluation_dataset.json"

# =====================================================
# MODEL SELECTION
# =====================================================

MODELS = {

    "gemma": "google/gemma-2-2b-it",

    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",

    "qwen": "Qwen/Qwen2.5-7B-Instruct"

}

# -----------------------------------------------------
# Change ONLY this line while comparing models
# -----------------------------------------------------

CURRENT_MODEL = "gemma"

MODEL_NAME = MODELS[CURRENT_MODEL]

# =====================================================
# BATCH SIZE (MODEL SPECIFIC)
# =====================================================

if CURRENT_MODEL == "gemma":

    BATCH_SIZE = 2

    GRADIENT_ACCUMULATION_STEPS = 8

else:

    # RTX 4050 (6GB) friendly settings

    BATCH_SIZE = 1

    GRADIENT_ACCUMULATION_STEPS = 16

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

BEST_MODEL_DIR = "best_models"

MODEL_SAVE_PATH = os.path.join(
    BEST_MODEL_DIR,
    CURRENT_MODEL
)

OUTPUT_DIR = os.path.join(
    "outputs",
    CURRENT_MODEL
)

METRICS_DIR = os.path.join(
    OUTPUT_DIR,
    "metrics"
)

LOG_DIR = os.path.join(
    "logs",
    CURRENT_MODEL
)

CHECKPOINT_DIR = os.path.join(
    "checkpoints",
    CURRENT_MODEL
)

# -----------------------------------------------------
# Create folders automatically
# -----------------------------------------------------

for directory in [

    BEST_MODEL_DIR,

    MODEL_SAVE_PATH,

    OUTPUT_DIR,

    METRICS_DIR,

    LOG_DIR,

    CHECKPOINT_DIR

]:

    os.makedirs(directory, exist_ok=True)

# =====================================================
# TRAINING
# =====================================================

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

MAX_SEQ_LENGTH = 512

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 1.0

# =====================================================
# QLoRA
# =====================================================

LOAD_IN_4BIT = True

BNB_4BIT_QUANT_TYPE = "nf4"

BNB_4BIT_USE_DOUBLE_QUANT = True

BNB_4BIT_COMPUTE_DTYPE = torch.float16

# =====================================================
# LoRA
# =====================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

BIAS = "none"

TASK_TYPE = "CAUSAL_LM"

# =====================================================
# CROSS VALIDATION
# =====================================================

NUM_FOLDS = 5

SHUFFLE = True

RANDOM_STATE = 42

# =====================================================
# EARLY STOPPING
# =====================================================

EARLY_STOPPING_PATIENCE = 2

EARLY_STOPPING_THRESHOLD = 0.0

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# CHECKPOINTS
# =====================================================

LOAD_BEST_MODEL_AT_END = True

SAVE_TOTAL_LIMIT = 2

SAVE_STRATEGY = "epoch"

EVALUATION_STRATEGY = "epoch"

LOGGING_STRATEGY = "epoch"

# =====================================================
# TENSORBOARD
# =====================================================

REPORT_TO = "tensorboard"

RUN_NAME = f"{CURRENT_MODEL}_5fold"

# =====================================================
# MODEL SAVING
# =====================================================

SAVE_SAFETENSORS = True

SAVE_ONLY_MODEL = False

# =====================================================
# PRINT CONFIGURATION
# =====================================================

print("\n" + "=" * 60)
print("Fine-Tuning Configuration Loaded")
print("=" * 60)
print(f"Current Model          : {CURRENT_MODEL}")
print(f"HuggingFace Model      : {MODEL_NAME}")
print(f"Device                 : {DEVICE}")
print(f"Epochs                 : {NUM_EPOCHS}")
print(f"Batch Size             : {BATCH_SIZE}")
print(f"Gradient Accumulation  : {GRADIENT_ACCUMULATION_STEPS}")
print(f"K-Folds                : {NUM_FOLDS}")
print(f"Learning Rate          : {LEARNING_RATE}")
print("=" * 60 + "\n")