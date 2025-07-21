import os
import uuid
import ray
from ray.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch

# 1) Model & I/O settings
HF_MODEL = "zhihan1996/DNABERT-2-117M"  # a genome foundation model :contentReference[oaicite:1]{index=1}
INPUT_PATH = os.environ.get("DNA_SEQ_INPUT", "s3://my-bucket/dna_sequences.csv")
OUTPUT_PATH = os.environ.get("EMBEDDING_OUTPUT", "s3://my-bucket/dna_embeddings/")

# 2) Startup Ray
if ray.is_initialized():
    ray.shutdown()
ray.init(runtime_env={"pip": ["transformers", "torch"]})

# 3) Load your DNA data as a Ray Dataset
#    Assumes a CSV with column "sequence"; adjust reader if using FASTA or plain text.
ds: Dataset = ray.data.read_csv(INPUT_PATH)

# 4) (Optional) k-mer chunking
def kmers(seq: str, k: int = 6, stride: int = 1):
    return [seq[i : i + k] for i in range(0, max(1, len(seq) - k + 1), stride)]

def chunk_row(row, seq_col="sequence"):
    pieces = kmers(row[seq_col], k=6, stride=3)
    return [
        {"id": str(uuid.uuid4()), seq_col: p, **{c: row[c] for c in row if c != seq_col}}
        for p in pieces
    ]

ds = ds.flat_map(chunk_row, fn_kwargs={"seq_col": "sequence"})

# 5) Define the embedding-UDF
class ComputeDNAEmbeddings:
    def __init__(self, seq_col: str, model_name: str, device: str = "cpu"):
        self.seq_col = seq_col
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def __call__(self, batch: dict):
        seqs = batch[self.seq_col]
        tokens = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.model(**tokens)
            # use [CLS] (first token) as fixed-length embedding
            embs = out.last_hidden_state[:, 0, :].cpu().numpy()
        return {
            **{self.seq_col: seqs},
            "embedding": list(embs),
            # keep any other metadata columns in batch...
        }

# 6) Run in parallel and write results
# TODO
    #num_gpus=1
    #
    #
embedded = ds.map_batches(
    ComputeDNAEmbeddings,
    batch_size=32,               # tune for memory
    fn_constructor_kwargs={
        "seq_col": "sequence",
        "model_name": HF_MODEL,
        "device": "cpu",         # or "cuda"
    },
    concurrency=4,               # number of actors
    num_gpus=0,                  # set >0 if using GPUs
)
embedded.write_parquet(OUTPUT_PATH, try_create_dir=True)
print("✅ Embeddings written to", OUTPUT_PATH)
