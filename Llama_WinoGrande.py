from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorForMultipleChoice
import transformers
import torch
from peft import LoraConfig,get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score
transformers.set_seed(42)
torch.manual_seed(42)
print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name())

dataset = load_dataset('/root/.cache/modelscope/hub/datasets/allenai___winogrande/default-d553340e27f46fdf')
train_data = dataset["train"].select(range(100))  # 仅使用前1000条数据
val_data = dataset["validation"].select(range(100))  # 仅使用前1000条数据

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B",use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def preprocess_function(examples):
    texts = []
    labels = []
    for i,sentence in enumerate(examples["sentence"]):
        option1_sentence = f"{sentence.replace("_",examples["option1"][i])}"
        option2_sentence = f"{sentence.replace("_",examples["option2"][i])}"
        text = (f"Evaluate the reasonableness of these completions:\n"
               f"Original: {sentence}\n"
               f"Option 1: {option1_sentence}\n"
               f"Option 2: {option2_sentence}")
        texts.append(text)

        correct_option = int(examples["answer"][i])
        label = correct_option - 1  # 将1,2转换为0,1
        labels.append(label)

    inputs = tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = labels
    return inputs

tokenized_train = train_data.map(preprocess_function,batched=True)
tokenized_val = val_data.map(preprocess_function,batched=True)

# tokenized_train.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
# tokenized_val.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-3B",
    num_labels=2,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.pad_token_id
training_args = TrainingArguments(
    output_dir = "./winog_model",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
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