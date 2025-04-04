import os
import time
import ray
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, default_data_collator
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, rank):
        self.rank = rank
        self.device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)
        self.tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased").to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=3e-5)
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"runs/ray_gpu_{rank}")
        print(f"rainer initialized on GPU {rank}")

    def preprocess(self, examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        inputs = self.tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384, return_tensors="pt")

        answers = examples["answers"]
        start_positions = [ans["answer_start"][0] if len(ans["answer_start"]) > 0 else 0 for ans in answers]
        end_positions = [start + len(ans["text"][0]) if len(ans["text"]) > 0 else 0 for start, ans in zip(start_positions, answers)]

        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)
        return inputs

    def train(self, epochs=3, batch_size=16):
        dataset = load_dataset("squad")
        train_data = dataset["train"].map(self.preprocess, batched=True, remove_columns=dataset["train"].column_names)
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)

        print(f"Starting training on GPU {self.rank}")
        self.model.train()
        start_time = time.time()

        for epoch in range(epochs):
            total_loss = 0
            for step, batch in enumerate(dataloader):
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                with autocast():
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf loss at step {step}, skipping.")
                    continue

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

                if step % 10 == 0:
                    alloc = torch.cuda.memory_allocated(self.device) / 1024**3
                    reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    avg_loss = total_loss / (step + 1)
                    print(f"[GPU {self.rank}] Epoch {epoch+1} | Step {step} | Loss: {avg_loss:.4f} | Mem: {alloc:.2f}/{reserved:.2f} GB")
                    self.writer.add_scalar("train_loss", avg_loss, epoch * len(dataloader) + step)
                    self.writer.add_scalar("gpu/memory_allocated_gb", alloc, epoch * len(dataloader) + step)
                    self.writer.add_scalar("gpu/memory_reserved_gb", reserved, epoch * len(dataloader) + step)

            self.writer.add_scalar("epoch_loss", total_loss / len(dataloader), epoch)

        elapsed = time.time() - start_time
        print(f"GPU {self.rank} training complete in {elapsed:.2f} seconds")
        self.writer.close()

        #Save model and tokenizer for each GPU
        output_dir = f"bert_squad_trained_ray/gpu_{self.rank}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"[GPU {self.rank}] Saving model and tokenizer to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[GPU {self.rank}] Model saved.")



if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    num_gpus = torch.cuda.device_count()
    print(f"Launching Ray training on {num_gpus} GPUs")

    workers = [Trainer.remote(rank=i) for i in range(num_gpus)]
    ray.get([worker.train.remote() for worker in workers])
    ray.shutdown()
