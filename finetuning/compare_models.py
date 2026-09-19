"""
=========================================================
Compare Fine-Tuned Models
Gemma | Mistral | Qwen
=========================================================
"""

import os
import json
import csv

# =====================================================
# MODELS
# =====================================================

MODELS = [

    "gemma",

    "mistral",

    "qwen"

]

# =====================================================
# PATHS
# =====================================================

OUTPUT_FOLDER = "outputs"

comparison_results = []

# =====================================================
# LOAD RESULTS
# =====================================================

print("\n")
print("=" * 70)
print("Comparing Fine-Tuned Models")
print("=" * 70)

for model in MODELS:

    file_path = os.path.join(

        OUTPUT_FOLDER,

        model,

        "metrics",

        "average_results.json"

    )

    if not os.path.exists(file_path):

        print(f"Skipping {model} (results not found)")

        continue

    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as f:

        data = json.load(f)

    comparison_results.append({

        "model": model,

        "bleu": data["average_bleu"],

        "rouge": data["average_rouge"],

        "cosine": data["average_cosine"]

    })

# =====================================================
# CHECK RESULTS
# =====================================================

if len(comparison_results) == 0:

    print("No evaluation results found.")

    exit()

# =====================================================
# FIND BEST MODEL
# =====================================================

best_model = max(

    comparison_results,

    key=lambda x: (

        x["bleu"] +

        x["rouge"] +

        x["cosine"]

    )

)

# =====================================================
# PRINT TABLE
# =====================================================

print("\n")

print("-" * 70)

print(f"{'Model':<15}{'BLEU':<15}{'ROUGE-L':<15}{'Cosine':<15}")

print("-" * 70)

for item in comparison_results:

    print(

        f"{item['model']:<15}"

        f"{item['bleu']:<15.4f}"

        f"{item['rouge']:<15.4f}"

        f"{item['cosine']:<15.4f}"

    )

print("-" * 70)

print(

    f"\nBest Model : {best_model['model']}"

)

print("=" * 70)

# =====================================================
# SAVE JSON
# =====================================================

json_file = os.path.join(

    OUTPUT_FOLDER,

    "comparison_results.json"

)

with open(

    json_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        {

            "results": comparison_results,

            "best_model": best_model

        },

        f,

        indent=4

    )

# =====================================================
# SAVE CSV
# =====================================================

csv_file = os.path.join(

    OUTPUT_FOLDER,

    "comparison_results.csv"

)

with open(

    csv_file,

    "w",

    newline="",

    encoding="utf-8"

) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Model",

        "BLEU",

        "ROUGE-L",

        "Cosine"

    ])

    for item in comparison_results:

        writer.writerow([

            item["model"],

            item["bleu"],

            item["rouge"],

            item["cosine"]

        ])

# =====================================================
# FINISHED
# =====================================================

print("\nResults saved successfully.")

print(f"JSON : {json_file}")

print(f"CSV  : {csv_file}")

print("=" * 70)