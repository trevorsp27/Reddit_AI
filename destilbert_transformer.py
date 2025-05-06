import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load your dataset
df = pd.read_csv("C:/Users/tspin/OneDrive/Desktop/reddit_AI/reddit_posts_labeled_clean.csv")

# Drop any rows with missing values in relevant columns
df = df.dropna(subset=['body', 'reliable'])

# Ensure 'reliable' is a clean boolean or integer
if df['reliable'].dtype == object:
    df['reliable'] = df['reliable'].astype(str).str.extract(r'(True|False)', expand=False)
    df['reliable'] = df['reliable'].map({'True': 1, 'False': 0})
else:
    df['reliable'] = df['reliable'].astype(int)

# Rename columns to match Hugging Face conventions
df = df.rename(columns={'body': 'text', 'reliable': 'label'})

# Convert to Dataset
hf_dataset = Dataset.from_pandas(df[['text', 'label']])

# Tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(tokenized_datasets["test"])
preds = predictions.predictions.argmax(axis=1)
print("\nClassification Report:")
print(classification_report(tokenized_datasets["test"]["label"], preds))
