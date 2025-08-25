from datasets import load_dataset
from transformers import BertTokenizer,DataCollatorForMultipleChoice,BertForMultipleChoice,TrainingArguments,Trainer,AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score
import torch
dataset =  load_dataset('/root/.cache/modelscope/hub/datasets/allenai___winogrande/default-d553340e27f46fdf')

train_data = dataset["train"]
val_data = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/google-bert/bert-base-uncased")

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

# tokenized_train = train_data.map(preprocess_function, batched=True).select(range(100))  # 仅使用前1000条数据
# tokenized_val = val_data.map(preprocess_function, batched=True).select(range(100))  # 仅使用前1000条数据
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_val = val_data.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    logits,labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits),dim=1)
    return {"accuracy":accuracy_score(labels, predictions)}

model = BertForMultipleChoice.from_pretrained(
    "/root/.cache/modelscope/hub/models/google-bert/bert-base-uncased",
    num_labels=2,
)

training_args = TrainingArguments(
    output_dir="my_awesome_swag_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

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
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

trainer.train()