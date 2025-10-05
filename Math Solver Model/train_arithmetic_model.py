# train_arithmetic_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Load dataset
df = pd.read_csv(r"MATH SOLVER MODEL\arithmetic_dataset.csv")

train_df, val_df = train_test_split(df, test_size=0.1)

# Tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

max_len = 32
def tokenize(batch):
    return tokenizer(batch['problem'], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

train_encodings = tokenize(train_df)
val_encodings = tokenize(val_df)

train_labels = tokenizer(list(train_df['solution']), padding='max_length', truncation=True, max_length=max_len, return_tensors="pt").input_ids
val_labels = tokenizer(list(val_df['solution']), padding='max_length', truncation=True, max_length=max_len, return_tensors="pt").input_ids

class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()} | {'labels': self.labels[idx]}

train_dataset = ArithmeticDataset(train_encodings, train_labels)
val_dataset = ArithmeticDataset(val_encodings, val_labels)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Save model
model.save_pretrained("./arithmetic_model")
tokenizer.save_pretrained("./arithmetic_model")
