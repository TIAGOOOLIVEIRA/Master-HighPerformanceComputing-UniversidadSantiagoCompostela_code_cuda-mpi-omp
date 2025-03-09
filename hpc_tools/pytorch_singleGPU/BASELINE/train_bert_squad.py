import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from datasets import load_dataset
from transformers import default_data_collator
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

#GPU Check
device = torch.device("CUDA" if torch.cuda.is_available() else "CPU")
print(f"Using device: {device}")

#Load Model & Tokenizer
model_name = "google-bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

dataset = load_dataset("squad")


def log_gpu_memory(step):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  #to GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)  #to GB
    print(f"📊 Step {step}: GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    writer.add_scalar("GPU/Allocated_Memory_GB", allocated, step)
    writer.add_scalar("GPU/Reserved_Memory_GB", reserved, step)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384, return_tensors="pt")

    #Convert answer start/end positions for loss calculation
    answers = examples["answers"]
    start_positions = [ans["answer_start"][0] if len(ans["answer_start"]) > 0 else 0 for ans in answers]
    end_positions = [start + len(ans["text"][0]) if len(ans["text"]) > 0 else 0 for start, ans in zip(start_positions, answers)]

    inputs["start_positions"] = torch.tensor(start_positions)
    inputs["end_positions"] = torch.tensor(end_positions)

    return inputs

#Apply Preprocessing
train_data = dataset["train"].map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=default_data_collator)

print("Dataset Loaded & Preprocessed")

#Define Optimizer & Loss
optimizer = AdamW(model.parameters(), lr=3e-5)
scaler = GradScaler()
criterion = nn.CrossEntropyLoss()

#TensorBoard Logger
log_dir = "runs/bert_squad_single_gpu"
writer = SummaryWriter(log_dir)

num_epochs = 3

def train(model, dataloader, optimizer, num_epochs=3):
    model.train()
    start_time = time.time()
    clip_grad = 1.0

    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            log_gpu_memory(batch_idx)
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**inputs)
                loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at step {batch_idx}, skipping update!")
                    continue  

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            #Log training loss every 10 steps
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                writer.add_scalar("Loss/train", avg_loss, epoch * len(dataloader) + batch_idx)
                print(f"Epoch {epoch+1} | Step {batch_idx} | Loss: {avg_loss:.4f}")

        #Log epoch loss
        writer.add_scalar("Loss/Epoch", total_loss / len(dataloader), epoch)

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

#Run Training
train(model, train_dataloader, optimizer, num_epochs)

#Save Model & Tokenizer
model.save_pretrained("bert_squad_trained")
tokenizer.save_pretrained("bert_squad_trained")
writer.close()

print("Training Completed & Model Saved")