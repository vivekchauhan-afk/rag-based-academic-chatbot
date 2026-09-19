"""
==========================================================
QLoRA Trainer
Supports:
    • Gemma
    • Mistral
    • Qwen

Compatible with:
    transformers 4.57.1
    trl 1.9.2
    peft 0.20.0
==========================================================
"""




from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    TaskType
)

from trl import (
    SFTTrainer,
    SFTConfig
)

from finetuning.callbacks import get_callbacks
from finetuning.dataset_loader import load_fold
from finetuning.config import *
# ==========================================================
# Check CUDA
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("Device :", DEVICE)

if DEVICE == "cuda":

    print("GPU :", torch.cuda.get_device_name(0))

print("=" * 60)

# ==========================================================
# Quantization Configuration
# ==========================================================
# ==========================================================
# Quantization Configuration
# ==========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.bfloat16,

    bnb_4bit_use_double_quant=True

)



# ==========================================================
# LoRA Target Modules
# ==========================================================

def get_target_modules(model_name):

    model_name = model_name.lower()

    if "gemma" in model_name:

        return [

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",

            "gate_proj",
            "up_proj",
            "down_proj"

        ]

    elif "mistral" in model_name:

        return [

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",

            "gate_proj",
            "up_proj",
            "down_proj"

        ]

    elif "qwen" in model_name:

        return [

            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",

            "gate_proj",
            "up_proj",
            "down_proj"

        ]

    else:

        raise ValueError("Unsupported model.")

# ==========================================================
# PEFT Configuration
# ==========================================================

def get_peft_config():

    return LoraConfig(

        task_type=TaskType.CAUSAL_LM,

        inference_mode=False,

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        target_modules=get_target_modules(
            MODEL_NAME
        )

    )

# ==========================================================
# Load Tokenizer
# ==========================================================

def load_tokenizer():

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(

        MODEL_NAME,

        trust_remote_code=True

    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Tokenizer Loaded.")

    return tokenizer

# ==========================================================
# Load Base Model
# ==========================================================

def load_model():

    print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        quantization_config=bnb_config,

        device_map={"": 0},
        torch_dtype=torch.bfloat16,

        trust_remote_code=True

    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    print("Model Loaded.")

    return model
# ==========================================================
# Training Configuration
# ==========================================================

def get_training_config(fold: int):

    config = SFTConfig(

        # -----------------------------
        # Output
        # -----------------------------

        output_dir=os.path.join(
            CHECKPOINT_DIR,
            f"fold_{fold}"
        ),

        overwrite_output_dir=True,

        # -----------------------------
        # Training
        # -----------------------------

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        warmup_ratio=WARMUP_RATIO,
        optim="paged_adamw_8bit",

        max_grad_norm=1.0,

        # -----------------------------
        # Evaluation
        # -----------------------------

        eval_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="eval_loss",

        greater_is_better=False,

        # -----------------------------
        # Logging
        # -----------------------------

        logging_dir=os.path.join(
            LOG_DIR,
            f"fold_{fold}"
        ),

        logging_strategy="epoch",

        report_to="tensorboard",

        save_total_limit=2,

        # -----------------------------
        # Mixed Precision
        # -----------------------------
        fp16=False,

        bf16=True,

        # -----------------------------
        # Dataset
        # -----------------------------

        dataset_text_field="text",

        max_length=MAX_SEQ_LENGTH,

        packing=False,

        # -----------------------------
        # Misc
        # -----------------------------

        gradient_checkpointing=True,

        seed=SEED

    )

    return config


# ==========================================================
# Create Trainer
# ==========================================================

def build_trainer(fold: int):

    print("\n")
    print("=" * 60)
    print(f"Preparing Fold {fold}")
    print("=" * 60)

    # ----------------------------------
    # Dataset
    # ----------------------------------

    train_dataset, valid_dataset = load_fold(fold)

    # ----------------------------------
    # Tokenizer
    # ----------------------------------

    tokenizer = load_tokenizer()

    # ----------------------------------
    # Model
    # ----------------------------------

    model = load_model()

    # ----------------------------------
    # LoRA
    # ----------------------------------

    peft_config = get_peft_config()

    # ----------------------------------
    # SFT Config
    # ----------------------------------

    training_config = get_training_config(fold)

    # ----------------------------------
    # Trainer
    # ----------------------------------

    trainer = SFTTrainer(

        model=model,


        args=training_config,

        train_dataset=train_dataset,

        eval_dataset=valid_dataset,

        processing_class=tokenizer,

        peft_config=peft_config,

        callbacks=get_callbacks()

    )

    print("=" * 60)
    print("Trainer Created Successfully")
    print("=" * 60)

    return trainer

# ==========================================================
# Train One Fold
# ==========================================================

def train_fold(fold: int):

    print("\n")
    print("=" * 70)
    print(f"Training Fold {fold}")
    print("=" * 70)

    trainer = build_trainer(fold)

    # ------------------------------------
    # Start Training
    # ------------------------------------

    train_result = trainer.train()

    # ------------------------------------
    # Evaluate
    # ------------------------------------

    eval_metrics = trainer.evaluate()

    # ------------------------------------
    # Save Best Model
    # ------------------------------------

    save_path = os.path.join(
        MODEL_SAVE_PATH,
        f"fold_{fold}"
    )

    os.makedirs(save_path, exist_ok=True)

    # Save model
    trainer.save_model(save_path)

    # Save tokenizer
    trainer.processing_class.save_pretrained(save_path)

    print(f"\nBest model saved to: {save_path}")

    # ------------------------------------
    # Save Trainer Metrics
    # ------------------------------------

    import json

    trainer_metrics_path = os.path.join(
        save_path,
        "trainer_metrics.json"
    )

    with open(
        trainer_metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            train_result.metrics,
            f,
            indent=4
        )

    # ------------------------------------
    # Save Training Metrics
    # ------------------------------------

    metrics = {

        "fold": fold,

        "train_loss": train_result.training_loss,

        "eval_loss": eval_metrics.get("eval_loss"),

        "runtime": train_result.metrics.get(
            "train_runtime",
            0
        ),

        "samples_per_second": train_result.metrics.get(
            "train_samples_per_second",
            0
        ),

        "epochs": NUM_EPOCHS,

        "learning_rate": LEARNING_RATE

    }

    metrics_path = os.path.join(
        save_path,
        "training_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(f"Metrics saved : {metrics_path}")

    # ------------------------------------
    # Free GPU Memory
    # ------------------------------------

    import gc

    del trainer
    del train_result
    del eval_metrics

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("GPU memory cleared.")

    # ------------------------------------
    # Return Metrics
    # ------------------------------------

    return metrics
# ==========================================================
# Train All Folds
# ==========================================================

def train_all_folds():

    results = []

    print("\n")
    print("=" * 70)
    print(f"Training Model : {MODEL_NAME}")
    print("=" * 70)

    # Resume from Fold 4
    for fold in range(4, NUM_FOLDS + 1):

        result = train_fold(fold)

        results.append(result)

    return results
# ==========================================================
# Main
# ==========================================================



if __name__ == "__main__":

    train_all_folds()



