# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B",device=1)
outputs = pipe("The key to life is", max_length=50, num_return_sequences=1)

print(outputs[0]['generated_text'])