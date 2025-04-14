import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForQuestionAnswering, BertTokenizer, default_data_collator
from transformers import AdamW
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384)

    start_positions = [
        ans["answer_start"][0] if len(ans["answer_start"]) > 0 else 0
        for ans in examples["answers"]
    ]
    end_positions = [
        start + len(ans["text"][0]) if len(ans["text"]) > 0 else 0
        for start, ans in zip(start_positions, examples["answers"])
    ]

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Logging
    if rank == 0:
        print(f"[Rank {rank}] Starting training on {world_size} GPUs.")
    start_time = time.time()
    writer = SummaryWriter(f"runs/bert_squad_ddp_rank{rank}")

    # Load tokenizer and dataset
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    dataset = load_dataset("squad")
    tokenized = dataset["train"].map(lambda x: preprocess_function(x, tokenizer),
                                     batched=True,
                                     remove_columns=dataset["train"].column_names)

    sampler = DistributedSampler(tokenized, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(tokenized,
                            sampler=sampler,
                            batch_size=8,
                            collate_fn=default_data_collator)

    model = BertForQuestionAnswering.from_pretrained("google-bert/bert-base-uncased").to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = AdamW(model.parameters(), lr=3e-5)

    model.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                print(f"[Rank {rank}] Step {step} | Loss: {loss.item():.4f} | "
                      f"Mem Alloc: {allocated:.2f} GB | Mem Res: {reserved:.2f} GB")
                writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)
                writer.add_scalar("GPU/Allocated_GB", allocated, step)
                writer.add_scalar("GPU/Reserved_GB", reserved, step)

    if rank == 0:
        model.module.save_pretrained("bert_squad_ddp_trained")
        tokenizer.save_pretrained("bert_squad_ddp_trained")
        print("[Rank 0] Model and tokenizer saved.")
        print(f"Total training time: {time.time() - start_time:.2f} seconds")

    writer.close()
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
