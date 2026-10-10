import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


class GemmaLLM:

    def __init__(
        self,
        base_model="google/gemma-2-2b-it",
        adapter_path="best_models/gemma/fold_5",
        max_new_tokens=100
    ):
        self.base_model_name = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens

        self._load_model()

    # =====================================================
    # LOAD MODEL
    # =====================================================

    def _load_model(self):

        print("Loading Gemma tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path
        )

        print("Loading Gemma model...")

        # -------------------------------------------------
        # 4-BIT QUANTIZATION
        # -------------------------------------------------

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        # -------------------------------------------------
        # LOAD BASE MODEL ON GPU
        # -------------------------------------------------

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.float16
        )

        # -------------------------------------------------
        # LOAD LoRA ADAPTER
        # -------------------------------------------------

        print("Loading LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path
        )

        self.model.eval()

        print("Gemma model ready.")
        print("Device:", next(self.model.parameters()).device)

    # =====================================================
    # GENERATE
    # =====================================================

    def generate(self, prompt: str) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        # Move input tensors to GPU
        inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Remove prompt tokens
        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()