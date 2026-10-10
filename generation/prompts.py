def build_rag_prompt(context: str, question: str) -> str:
    """
    Build the prompt used by the RAG system.

    The LLM must answer only from the retrieved context.
    """

    prompt = f"""
You are an AI Assistant for National Institute of Technology Kurukshetra.

Your task is to answer the question ONLY from the Retrieved Context.

STRICT RULES:

1. Do not use pretrained knowledge.
2. Do not use information from outside the Retrieved Context.
3. Do not guess or hallucinate.
4. For names, dates, numbers, designations, and official positions,
   use the exact information stated in the Retrieved Context.
5. If the Retrieved Context contains a person explicitly associated
   with the requested position, return that person's name.
6. Do not replace the retrieved person's name with another person.
7. If the answer cannot be found in the Retrieved Context, say exactly:

"I could not find this information in the available documents."

Retrieved Context:
{context}

Question:
{question}

Answer:
"""

    return prompt