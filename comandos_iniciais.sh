python -m venv llm_tradeoff
source llm_tradeoff/bin/activate  # Linux/MacOS
# ou llm_tradeoff\Scripts\activate  # Windows

pip install torch transformers datasets peft accelerate huggingface_hub deepeval==0.21.0 pytest