import os
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import BertForQuestionAnswering, BertTokenizer
from torch.optim import AdamW

from datasets import load_dataset
from transformers import default_data_collator
from pytorch_lightning.loggers import TensorBoardLogger


#this script is just a tester
#to validate main capabilities can run with a single GPU
#The motivation for this tester is due to the fact that the main script is too long and complex
#and it is difficult make the HPC environment stable enough with libraries dependencies 

#install
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install pytorch-lightning transformers datasets tensorboard

#to run
#python3 -m venv venv
#source venv/bin/activate

#to launch
#python train_bert_squad_single_gpu.py

#logs
#tensorboard --logdir lightning_logs



#GPU Availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load Model & Tokenizer
model_name = "google-bert/bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

#Load Dataset
dataset = load_dataset("squad")

#Preprocessing Function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    inputs = tokenizer(questions, contexts, truncation=True, padding="max_length", max_length=384, return_tensors="pt")

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

#Define PyTorch Lightning Module
class BertLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.lr = 3e-5

    def on_train_start(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"🚀 Training on {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
            self.logger.experiment.add_text("Training Info", f"Using {num_gpus} GPUs")

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        #GPU Memory Monitoring (Optional)
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            allocated_memory = torch.cuda.memory_allocated(current_gpu) / (1024 ** 3)
            reserved_memory = torch.cuda.memory_reserved(current_gpu) / (1024 ** 3)
            print(f"Step {batch_idx} | GPU {current_gpu} | Allocated: {allocated_memory:.2f} GB | Reserved: {reserved_memory:.2f} GB")
            self.logger.experiment.add_scalar(f"GPU_{current_gpu}/Allocated_Memory_GB", allocated_memory, batch_idx)
            self.logger.experiment.add_scalar(f"GPU_{current_gpu}/Reserved_Memory_GB", reserved_memory, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

#Define TensorBoard Logger and Trainer
logger = TensorBoardLogger("lightning_logs", name="bert_single_gpu")
trainer = pl.Trainer(
    max_epochs=3,
    accelerator="gpu",     #Use GPU if available
    devices=1,             
    precision=16,          #precision for speed
    strategy="auto",       #Lightning choose best - no FSDP
    logger=logger,
)

#Train
model = BertLightningModel()
trainer.fit(model, train_dataloader)

#Save Model and Tokenizer
if trainer.is_global_zero:
    print("Saving model and tokenizer")
    model.model.save_pretrained("bert_squad_trained")
    tokenizer.save_pretrained("bert_squad_trained")
    print("Model saved to ./bert_squad_trained")
