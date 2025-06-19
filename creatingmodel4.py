import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load dataset from CSV file
df = pd.read_csv("indian_depression_dataset.csv")  # Ensure the CSV is in the same directory

# Check dataset structure
print(df.head())  # Optional: Print first few rows to verify loading
# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['statement'], df['depression_level'], test_size=0.2, random_state=42
)

# Tokenizer and model initialization
# Using ALBERT-base-v2
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=4)
# Tokenize the data
def tokenize_data(texts, labels):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128)
    return Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })

train_dataset = tokenize_data(train_texts, list(train_labels))
val_dataset = tokenize_data(val_texts, list(val_labels))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./depression_detector4")

# Test the model
def predict(statement):
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Example usage
print(predict("I feel stressed and helpless about my future."))
