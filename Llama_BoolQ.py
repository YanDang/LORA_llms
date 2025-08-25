from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
import torch
from peft import LoraConfig,get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score

print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name())

dataset = load_dataset("/root/.cache/huggingface/hub/datasets--google--boolq/snapshots/35b264d03638db9f4ce671b711558bf7ff0f80d5/")
train_data = dataset["train"]
val_data = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B",use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Classification task, so we need to preprocess the data
def preprocess_function(examples):
    # texts = [
    #     f"Question: {q} Answer based on: {p}"
    #     for q, p in zip(examples["question"], examples["passage"])
    # ]
    # inputs = tokenizer(
    #     texts,
    #     max_length=512,
    #     truncation=True,
    #     padding="max_length"
    # )
    # inputs["labels"] = [int(label) for label in examples["answer"]]
    # return inputs
    inputs = tokenizer(
                examples['question'],
                examples['passage'],
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
    inputs["labels"] = [int(label) for label in examples["answer"]]
    return inputs

tokenized_train = train_data.map(preprocess_function,batched=True)
tokenized_val = val_data.map(preprocess_function,batched=True)

tokenized_train.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
tokenized_val.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B",
    num_labels=2,
    device_map="auto"
)

model.config.pad_token_id = tokenizer.pad_token_id
training_args = TrainingArguments(
    output_dir = "./boolq_model",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps = 8,
    num_train_epochs = 5,
    weight_decay = 0.01,
    save_strategy = "epoch",
    fp16=True,
    dataloader_pin_memory=False
)

def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy":accuracy_score(labels, predictions)}



lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model,lora_config)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_val,
    compute_metrics = compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)