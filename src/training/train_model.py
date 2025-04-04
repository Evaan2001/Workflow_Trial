import torch
import numpy as np
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


def train_classifier(
    model_name="distilbert-base-uncased",
    dataset_name="glue",
    dataset_config="sst2",
    output_dir=None,
    batch_size=16,
    learning_rate=5e-5,
    epochs=1,
):
    if output_dir is None:
        output_dir = os.path.join(
            "~/.cache/huggingface/hub", model_name.replace("/", "_")
        )

    # Expand user path if needed
    if output_dir and output_dir.startswith("~"):
        output_dir = os.path.expanduser(output_dir)

    # model_dir = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created model directory at: {output_dir}")

    # Create checkpoints subdirectory
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Created checkpoints directory at: {checkpoints_dir}")

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    print(f"Finished loading dataset: {dataset_name}!")
    dataset["train"] = dataset["train"].select(range(32))
    dataset["validation"] = dataset["validation"].select(range(32))
    print("Using 32-sample subsets for testing!")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Finished loading tokenizer!")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print("Finished loading model!")

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("Finished tokenizing dataset!")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=checkpoints_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    print("Finished setting up training arguments!")

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    print("Finished setting up trainer instance!")

    # Train model
    trainer.train()
    print("Finished training the model!")

    # After setting up the trainer
    print(
        "------------------------------Model architecture------------------------------"
    )
    print(f"Model type: {trainer.model.__class__.__name__}")

    # Get parameter counts
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # After training and before saving, run an inference test
    test_text = "This movie was really good!"
    print(f"\n----- Testing inference before saving -----")
    print(f"Test text: '{test_text}'")

    # Get the device the model is currently on
    # retrieves the device of the first parameter tensor in your model.
    # This gives you the exact device where your model currently resides.
    device = next(model.parameters()).device
    # print(f"Model is on device: {device}")

    # Ensure inputs are on the same device
    inputs = tokenizer(test_text, return_tensors="pt")
    inputs = {
        k: v.to(device) for k, v in inputs.items()
    }  # moves all your input tensors to that same device.
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    print(f"Predicted class before saving: {predicted_class}")
    print("----------------------------------------\n")

    # Save model
    # model_save_path = os.path.join(output_dir, model_name.replace("/", "_"))
    # os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Finished saving model related files!")

    return output_dir


if __name__ == "__main__":

    # Detect if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        # Print NVIDIA GPU information
        print(f"NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Train the model
    # cache_dir = "~/.cache/huggingface/hub"
    output_path = train_classifier()
    print(f"Model saved to {output_path}")

    # Load the saved model to verify
    print("Loading saved model to verify...")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(output_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_path)

    # Count parameters
    total_params = sum(p.numel() for p in loaded_model.parameters())
    trainable_params = sum(
        p.numel() for p in loaded_model.parameters() if p.requires_grad
    )

    print(f"Verification - Loaded model parameters: {total_params:,}")
    print(f"Verification - Loaded trainable parameters: {trainable_params:,}")

    # Optional: Try a simple inference to confirm everything works
    test_text = "This movie was really good!"
    inputs = loaded_tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Test inference on '{test_text}' - Predicted class: {predicted_class}")
