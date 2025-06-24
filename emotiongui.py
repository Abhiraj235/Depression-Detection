import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Load the pretrained model and tokenizer model 1
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("./depression_detector")
#Below are the three model we can uncomment it 
# Load the pretrained model and tokenizer model 2
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model = AutoModelForSequenceClassification.from_pretrained("./depression_detector2")

# Load the pretrained model and tokenizer model 3
#tokenizer = AutoTokenizer.from_pretrained("roberta-base")
#model = AutoModelForSequenceClassification.from_pretrained("./depression_detector3")

# Load the pretrained model and tokenizer model 3
#tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
#model = AutoModelForSequenceClassification.from_pretrained("./depression_detector4")


# Function to predict depression level
def predict(statement):
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    levels = {0: " Stress-Free", 1: "Low Stress", 2: "Medium Stress", 3: "High Stress"}
    return levels[prediction]

# GUI setup
def create_gui():
    def analyze_text():
        user_input = text_entry.get("1.0", tk.END).strip()
        if not user_input:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return
        result = predict(user_input)
        result_label.config(text=f"Depression Level: {result}")

    # Main window
    root = tk.Tk()
    root.title("Depression Level Detector")
    root.geometry("500x300")

    # Title label
    title_label = tk.Label(root, text="Depression Level Detector", font=("Helvetica", 16, "bold"), pady=10)
    title_label.pack()

    # Input text box
    text_frame = tk.Frame(root)
    text_frame.pack(pady=10)
    text_label = tk.Label(text_frame, text="Enter your text below:", font=("Helvetica", 12))
    text_label.pack(anchor="w")
    text_entry = tk.Text(text_frame, wrap="word", width=50, height=5, font=("Helvetica", 12))
    text_entry.pack()

    # Analyze button
    analyze_button = ttk.Button(root, text="Analyze", command=analyze_text)
    analyze_button.pack(pady=10)

    # Result label
    result_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"), fg="blue")
    result_label.pack(pady=10)

    # Start the GUI loop
    root.mainloop()
if __name__ == "__main__":
    create_gui()
