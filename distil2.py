import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight # Re-add this import

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
df = pd.read_csv("C:/Users/tspin/OneDrive/Desktop/reddit_AI/reddit_posts_labeled_clean.csv")

# Drop rows with missing title, body, or reliable values
df = df.dropna(subset=["title", "body", "reliable"])

# Fix any weird formatting in 'reliable' column (e.g., stringified objects)
df['reliable'] = df['reliable'].astype(str).str.extract(r'(True|False)', expand=False).map({'True': 1, 'False': 0})

# Drop rows with invalid labels
df = df[df['reliable'].isin([0, 1])]
df['label'] = df['reliable'].astype(int)

# Combine title and body
df['text'] = df['title'].astype(str) + " " + df['body'].astype(str)

# Split into train/test using the full dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), # Use full 'text' column
    df["label"].tolist(), # Use full 'label' column
    test_size=0.2,
    stratify=df["label"], # Stratify based on full labels
    random_state=42
)

# Tokenize using BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Convert to torch dataset
class RedditDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = RedditDataset(train_encodings, train_labels)
test_dataset = RedditDataset(test_encodings, test_labels)

# Manually set class weights
class_weights = torch.tensor([0.8, 1.3], dtype=torch.float)

print(f"Using Manual Class Weights: {class_weights}")

# Define WeightedTrainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None): # Add num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Move class_weights to the same device as logits
        weights = class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch", # Evaluate every epoch
    save_strategy="epoch",       # Save checkpoint every epoch
    load_best_model_at_end=True, # Load the best model based on evaluation
    metric_for_best_model="eval_loss", # Use eval loss to determine the best model
    greater_is_better=False      # Lower eval loss is better
)

# Initialize Trainer with WeightedTrainer
trainer = WeightedTrainer( # Use the custom WeightedTrainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer, # Pass tokenizer
    # class_weights are now handled inside WeightedTrainer's compute_loss
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(test_labels, predicted_labels))

# Plot confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_distil_manual_08_13.png") # Save the plot
print("Confusion matrix saved to confusion_matrix_distil_manual_08_13.png")
# plt.show() # Optionally display the plot
