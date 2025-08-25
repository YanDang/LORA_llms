from datasets import load_dataset
from transformers import BertTokenizer,BertForSequenceClassification,TrainingArguments,Trainer
import torch
from peft import LoraConfig,get_peft_model
from sklearn.metrics import accuracy_score

dataset = load_dataset("/root/.cache/huggingface/hub/datasets--google--boolq/snapshots/35b264d03638db9f4ce671b711558bf7ff0f80d5/")
train_data = dataset["train"]
val_data = dataset["validation"]

tokenizer = BertTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/google-bert/bert-base-uncased")

# Classification task, so we need to preprocess the data
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["passage"],
        max_length=256,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt"
    )
    # inputs["labels"] = examples["answer"]
    inputs["labels"] = [int(label) for label in examples["answer"]]
    return inputs

tokenized_train = train_data.map(preprocess_function,batched=True)
tokenized_val = val_data.map(preprocess_function,batched=True)

tokenized_train.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
tokenized_val.set_format(type="torch",columns=["input_ids","attention_mask","labels"])

label2id = {"False": 0, "True": 1}
id2label = {0: "False", 1: "True"}

model = BertForSequenceClassification.from_pretrained(
    "/root/.cache/modelscope/hub/models/google-bert/bert-base-uncased",
    num_labels=2,
    label2id=label2id,
    id2label=id2label,
    device_map="cuda:1"
)

training_args = TrainingArguments(
    output_dir = "./boolq_model",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,
    weight_decay = 0.01,
    save_strategy = "epoch",
    label_names=["False", "True"]  # Specify label names for evaluation
)

def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy":accuracy_score(labels, predictions)}



lora_config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = ['query','value'],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "SEQ_CLS"
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