from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorForMultipleChoice,AutoModelForMultipleChoice
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
    first_sentences = [[sentence] * 2 for sentence in examples["sentence"]]
    second_sentences = []
    for i,sentence in enumerate(examples["sentence"]):
        option1_sentence = sentence.replace("_",examples["option1"][i])
        option2_sentence = sentence.replace("_",examples["option2"][i])
        second_sentences.append([option1_sentence,option2_sentence])

    first_sentences = sum(first_sentences,[])
    second_sentences = sum(second_sentences,[])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding=True,
        max_length=512
    )
    result = {k: [v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    result['labels'] = [int(label) - 1 for label in examples["answer"]]
    return result

tokenized_train = train_data.map(preprocess_function,batched=True)
tokenized_val = val_data.map(preprocess_function,batched=True)

# tokenized_train.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
# tokenized_val.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
model = AutoModelForMultipleChoice.from_pretrained(
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
    data_collator=collator,
    compute_metrics = compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)