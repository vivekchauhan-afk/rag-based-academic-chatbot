"""
=========================================================
Evaluate Fine-Tuned Models
Gemma | Mistral | Qwen
5-Fold Cross Validation
=========================================================
"""


import json


from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

from datasets import load_from_disk

from finetuning.metrics import (
    compute_bleu,
    compute_rouge,
    compute_cosine,
    compute_bertscore,
    compute_semantic,
    compute_semantic_accuracy
)
from finetuning.config import *

# =====================================================
# LOAD TOKENIZER
# =====================================================

def load_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# =====================================================
# LOAD BASE MODEL
# =====================================================

# =====================================================
# LOAD BASE MODEL
# =====================================================

def load_base_model():

    base_model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        device_map={"": 0},

        torch_dtype=torch.bfloat16

    )

    return base_model

# =====================================================
# LOAD FINE-TUNED MODEL
# =====================================================

def load_model(fold):

    print("\n" + "=" * 60)
    print(f"Loading Fold {fold} Model")
    print("=" * 60)

    tokenizer = load_tokenizer()

    base_model = load_base_model()

    model_path = os.path.join(

        MODEL_SAVE_PATH,

        f"fold_{fold}"

    )

    model = PeftModel.from_pretrained(

        base_model,

        model_path

    )

    model.eval()

    print("Model Loaded Successfully.")

    return tokenizer, model


# =====================================================
# LOAD VALIDATION DATA
# =====================================================

def load_validation_data(fold):

    dataset_path = os.path.join(

        "finetuning",

        "folds",

        f"fold_{fold}_valid"

    )

    dataset = load_from_disk(
        dataset_path
    )

    print(f"Validation Samples : {len(dataset)}")

    return dataset


# =====================================================
# GENERATE MODEL RESPONSE
# =====================================================

def generate_answer(

    model,

    tokenizer,

    question

):

    prompt = f"""### Instruction:

{question}

### Response:
"""

    inputs = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_SEQ_LENGTH

    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=256,

            do_sample=False,

            temperature=0.0

        )

    prediction = tokenizer.decode(

        outputs[0],

        skip_special_tokens=True

    )

    if "### Response:" in prediction:

        prediction = prediction.split(
            "### Response:"
        )[-1].strip()

    return prediction

# =====================================================
# EVALUATE SINGLE FOLD
# =====================================================

# =====================================================
# EVALUATE SINGLE FOLD
# =====================================================

def evaluate_fold(fold):

    tokenizer, model = load_model(fold)

    validation_dataset = load_validation_data(fold)

    predictions = []

    bleu_scores = []
    rouge_scores = []
    cosine_scores = []
    bert_scores = []
    semantic_scores = []
    semantic_accuracy_scores = []

    print(f"\nEvaluating Fold {fold}...\n")

    for sample in tqdm(validation_dataset):

        question = sample["question"]

        gold_answer = sample["answer"]

        prediction = generate_answer(

            model,

            tokenizer,

            question

        )

        # ----------------------------------------
        # Compute Metrics
        # ----------------------------------------

        bleu = compute_bleu(
            gold_answer,
            prediction
        )

        rouge = compute_rouge(
            gold_answer,
            prediction
        )

        cosine = compute_cosine(
            gold_answer,
            prediction
        )

        bert = compute_bertscore(
            gold_answer,
            prediction
        )

        semantic = compute_semantic(
            gold_answer,
            prediction
        )

        semantic_accuracy = compute_semantic_accuracy(
            gold_answer,
            prediction
        )

        # ----------------------------------------
        # Store Scores
        # ----------------------------------------

        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
        cosine_scores.append(cosine)
        bert_scores.append(bert)
        semantic_scores.append(semantic)
        semantic_accuracy_scores.append(
            semantic_accuracy
        )

        # ----------------------------------------
        # Save Prediction
        # ----------------------------------------

        predictions.append({

            "question": question,

            "gold_answer": gold_answer,

            "prediction": prediction,

            "bleu": bleu,

            "rouge": rouge,

            "cosine": cosine,

            "bertscore": bert,

            "semantic": semantic,

            "semantic_accuracy": semantic_accuracy

        })

    # ------------------------------------------
    # Average Metrics
    # ------------------------------------------

    fold_result = {

        "fold": fold,

        "bleu":
            sum(bleu_scores) /
            max(len(bleu_scores), 1),

        "rouge":
            sum(rouge_scores) /
            max(len(rouge_scores), 1),

        "cosine":
            sum(cosine_scores) /
            max(len(cosine_scores), 1),

        "bertscore":
            sum(bert_scores) /
            max(len(bert_scores), 1),

        "semantic":
            sum(semantic_scores) /
            max(len(semantic_scores), 1),

        "semantic_accuracy":
            (
                sum(semantic_accuracy_scores)
                /
                max(len(semantic_accuracy_scores), 1)
            ) * 100

    }

    # ------------------------------------------
    # Save Predictions
    # ------------------------------------------

    prediction_file = os.path.join(

        METRICS_DIR,

        f"fold_{fold}_predictions.json"

    )

    with open(

        prediction_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            predictions,

            f,

            indent=4,

            ensure_ascii=False

        )

    # ------------------------------------------
    # Save Fold Metrics
    # ------------------------------------------

    fold_file = os.path.join(

        METRICS_DIR,

        f"fold_{fold}.json"

    )

    with open(

        fold_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            fold_result,

            f,

            indent=4

        )

    # ------------------------------------------
    # Print Results
    # ------------------------------------------

    print("\n")
    print("-" * 60)
    print(f"Fold {fold} Completed")
    print(f"BLEU                : {fold_result['bleu']:.4f}")
    print(f"ROUGE-L             : {fold_result['rouge']:.4f}")
    print(f"Cosine Similarity   : {fold_result['cosine']:.4f}")
    print(f"BERTScore           : {fold_result['bertscore']:.4f}")
    print(f"Semantic Similarity : {fold_result['semantic']:.4f}")
    print(f"Semantic Accuracy   : {fold_result['semantic_accuracy']:.2f}%")
    print("-" * 60)

    return fold_result
# =====================================================
# MAIN EVALUATION
# =====================================================

def main():

    print("\n")
    print("=" * 70)
    print("Fine-Tuned Model Evaluation")
    print("=" * 70)
    print(f"Model : {CURRENT_MODEL}")
    print("=" * 70)

    all_results = []

    # ------------------------------------------
    # Evaluate All Folds
    # ------------------------------------------

    for fold in range(1, NUM_FOLDS + 1):

        fold_result = evaluate_fold(fold)

        all_results.append(fold_result)

    # ------------------------------------------
    # Compute Average Metrics
    # ------------------------------------------

    average_results = {

        "model": CURRENT_MODEL,

        "model_name": MODEL_NAME,

        "num_folds": NUM_FOLDS,

        "average_bleu":

            sum(
                x["bleu"]
                for x in all_results
            ) / len(all_results),

        "average_rouge":

            sum(
                x["rouge"]
                for x in all_results
            ) / len(all_results),

        "average_cosine":

            sum(
                x["cosine"]
                for x in all_results
            ) / len(all_results),

        "average_bertscore":

            sum(
                x["bertscore"]
                for x in all_results
            ) / len(all_results),

        "average_semantic":

            sum(
                x["semantic"]
                for x in all_results
            ) / len(all_results),

        "average_semantic_accuracy":

            sum(
                x["semantic_accuracy"]
                for x in all_results
            ) / len(all_results)

    }
    # ------------------------------------------
    # Save Average Results
    # ------------------------------------------

    average_file = os.path.join(

        METRICS_DIR,

        "average_results.json"

    )

    with open(

        average_file,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            average_results,

            f,

            indent=4

        )

    # ------------------------------------------
    # Display Results
    # ------------------------------------------

    print("\n")
    print("=" * 70)
    print("Evaluation Completed Successfully")
    print("=" * 70)
    print(f"Model                 : {CURRENT_MODEL}")
    print(f"BLEU                  : {average_results['average_bleu']:.4f}")
    print(f"ROUGE-L               : {average_results['average_rouge']:.4f}")
    print(f"Cosine Similarity     : {average_results['average_cosine']:.4f}")
    print(f"BERTScore             : {average_results['average_bertscore']:.4f}")
    print(f"Semantic Similarity   : {average_results['average_semantic']:.4f}")
    print(f"Semantic Accuracy     : {average_results['average_semantic_accuracy']:.2f}%")

    print("=" * 70)
    print(f"Results Saved : {average_file}")
    print("=" * 70)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()