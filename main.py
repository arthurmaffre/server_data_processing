import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from transformers import AutoModel
from transformers import AutoTokenizer

import os
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

import math


def download_dataset(repo_id_download: str, filename: str = "pre_dataset.parquet", private: bool = False):
    """
    Download a dataset file from the Hugging Face Hub and load it into a pandas DataFrame.
    
    Args:
        repo_id (str): Hugging Face repo ID (e.g., "Arthurmaffre34/pre-dataset").
        filename (str): The file name inside the repo (default: "pre_dataset.parquet").
        private (bool): Set to True if the repo is private (requires HF_TOKEN).
    
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    file_path = hf_hub_download(
        repo_id=repo_id_download,
        filename=filename,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN") if private else None
    )
    return pd.read_parquet(file_path)

def reduce_dataset(df: pd.DataFrame, frac: float = None, n: int = None, seed: int = 42) -> pd.DataFrame:
    """
    Reduce the dataset by sampling either a fraction or a fixed number of rows.
    
    Args:
        df (pd.DataFrame): Input dataset.
        frac (float, optional): Fraction of the dataset to sample (e.g., 0.01 for 1%).
        n (int, optional): Exact number of rows to sample.
        seed (int): Random seed for reproducibility (default: 42).
    
    Returns:
        pd.DataFrame: Reduced dataset.
    """
    if frac is not None:
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    elif n is not None:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        raise ValueError("You must specify either `frac` or `n`.")

class FinBertEmbeddingDataset(Dataset):
    def __init__(self, df, tokenizer, model, max_length=512, n_chunks=100, sub_batch_size=100, device="cpu"):
        # --- Error checks ---
        if max_length > 512:
            raise ValueError(f"❌ max_length={max_length} is not allowed. FinBERT only supports max_length ≤ 512.")
        if sub_batch_size > n_chunks:
            raise ValueError(f"❌ sub_batch_size={sub_batch_size} cannot be greater than n_chunks={n_chunks}.")

        # --- Tokenizer check ---
        expected_model_name = "yiyanghkust/finbert-pretrain"
        if getattr(tokenizer, "name_or_path", None) != expected_model_name:
            warnings.warn(
                f"⚠️ Tokenizer `{tokenizer.name_or_path}` does not match expected `{expected_model_name}`. "
                "This may cause mismatches between embeddings and vocabulary.",
                UserWarning
            )

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.max_length = max_length
        self.n_chunks = n_chunks
        self.pad_id = tokenizer.pad_token_id
        self.sub_batch_size = sub_batch_size
        self.device = device
        self.model.eval()

    def chunk_text(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        chunks = [tokens[i:i+self.max_length] for i in range(0, len(tokens), self.max_length)]

        ids, masks = [], []
        for chunk in chunks[:self.n_chunks]:
            attn = [1] * len(chunk)
            if len(chunk) < self.max_length:
                pad_len = self.max_length - len(chunk)
                chunk = chunk + [self.pad_id] * pad_len
                attn = attn + [0] * pad_len
            ids.append(chunk)
            masks.append(attn)

        while len(ids) < self.n_chunks:
            ids.append([self.pad_id] * self.max_length)
            masks.append([0] * self.max_length)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def encode_chunks(self, ids, mask, pbar=None):
        outputs = []
        with torch.no_grad():
            for i in range(0, ids.size(0), self.sub_batch_size):
                ids_sub = ids[i:i+self.sub_batch_size].to(self.device)
                mask_sub = mask[i:i+self.sub_batch_size].to(self.device)
                with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                    out = self.model(ids_sub, attention_mask=mask_sub).pooler_output
                outputs.append(out.cpu())

                if pbar is not None:
                    pbar.update(ids_sub.size(0))  # global increment
        return torch.cat(outputs, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rf_ids, rf_mask = self.chunk_text(str(row["rf"]))
        mgmt_ids, mgmt_mask = self.chunk_text(str(row["mgmt"]))

        rf_emb = self.encode_chunks(rf_ids, rf_mask, pbar=getattr(self, "pbar", None))
        mgmt_emb = self.encode_chunks(mgmt_ids, mgmt_mask, pbar=getattr(self, "pbar", None))

        stacked = torch.stack([rf_emb, mgmt_emb], dim=0)

        return {
            "embeddings": stacked,
            "labels": torch.tensor(row["return"], dtype=torch.float)
        }

    def __len__(self):
        return len(self.df)
    

def prepare_dataset(df, tokenizer, model, sub_batch_size=32, device="cpu", n_chunks=100, batch_size=1):
    """
    Encode a dataset into FinBERT embeddings with progress bar.
    
    Args:
        df (pd.DataFrame): Input dataframe with "rf", "mgmt", and "return".
        tokenizer: Hugging Face tokenizer (e.g., FinBERT tokenizer).
        model: Hugging Face model (e.g., FinBERT model).
        sub_batch_size (int): Number of chunks processed at once inside FinBERT.
        device (str): "cpu" or "cuda".
        n_chunks (int): Number of chunks per document.
        batch_size (int): DataLoader batch size (default=1).
    
    Returns:
        (torch.Tensor, torch.Tensor): embeddings [num_samples, 2, N, hidden_dim], labels [num_samples]
    """
    dataset = FinBertEmbeddingDataset(
        df, tokenizer, model,
        sub_batch_size=sub_batch_size,
        device=device,
        n_chunks=n_chunks
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_chunks = len(dataset) * 2 * dataset.n_chunks  # RF + MGMT
    all_embeddings, all_labels = [], []

    with tqdm(total=total_chunks, desc="Encoding all chunks") as pbar:
        dataset.pbar = pbar
        for batch in loader:
            all_embeddings.append(batch["embeddings"])
            all_labels.append(batch["labels"])
        dataset.pbar = None  # cleanup

    embeddings = torch.cat(all_embeddings)   # [num_samples, 2, N, hidden_dim]
    labels = torch.cat(all_labels)           # [num_samples]
    return embeddings, labels

def save_sharded_dataset(embeddings, labels, max_file_size_gb=1, prefix="finbert_embeddings_part"):
    """
    Save embeddings + labels into multiple shards based on target file size.

    Args:
        embeddings (torch.Tensor): Tensor of shape [N, ...].
        labels (torch.Tensor): Tensor of shape [N].
        max_file_size_gb (int): Maximum shard size in gigabytes (default=1 GB).
        prefix (str): Prefix for saved shard files.

    Returns:
        int: Number of shards created.
    """
    # Estimate bytes per sample
    bytes_per_sample = (
        embeddings[0].element_size() * embeddings[0].numel() +
        labels[0].element_size()
    )
    shard_size = max(1, int((max_file_size_gb * (1024**3)) // bytes_per_sample))

    n_samples = len(embeddings)
    n_shards = math.ceil(n_samples / shard_size)

    print(f"📦 Target max file size: {max_file_size_gb} GB")
    print(f"⚖️  Estimated {bytes_per_sample/1024:.2f} KB per sample")
    print(f"➡️  {shard_size} samples per shard → {n_shards} shards total")

    for i in range(n_shards):
        start = i * shard_size
        end = min((i+1) * shard_size, n_samples)
        filename = f"{prefix}{i}.pt"
        torch.save(
            {"embeddings": embeddings[start:end], "labels": labels[start:end]},
            filename
        )
        file_size_mb = os.path.getsize(filename) / (1024**2)
        print(f"✅ Saved {filename} ({file_size_mb:.2f} MB, {end-start} samples)")

    return n_shards

def upload_dataset_shards_to_hf(repo_id_upload, n_shards, prefix="finbert_embeddings_part", repo_type="dataset"):
    from huggingface_hub import HfApi
    api = HfApi()
    for i in range(n_shards):
        filename = f"{prefix}{i}.pt"
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id_upload,
            repo_type=repo_type,
            create_pr=True
        )
        print(f"🔼 Upload {filename} vers HF terminé")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBERT embedding pipeline")
    
    # Input / output repos
    parser.add_argument("--repo_id_download", type=str, default="Arthurmaffre34/pre-dataset",
                        help="Hugging Face repo ID to download dataset from")
    parser.add_argument("--repo_id_upload", type=str, default="Gill-Hack-25-UdeM/Text_Embedding_Dataset",
                        help="Hugging Face repo ID to upload shards to")

    # Model / batching params
    parser.add_argument("--sub_batch_size", type=int, default=10, help="Chunks per forward pass in FinBERT")
    parser.add_argument("--n_chunks", type=int, default=100, help="Max number of chunks per document")
    parser.add_argument("--max_file_size_gb", type=float, default=0.5, help="Shard file size in GB")
    parser.add_argument("--sample_n", type=int, default=None, help="Reduce dataset to N rows (debug mode)")
    parser.add_argument("--sample_frac", type=float, default=None, help="Reduce dataset to a fraction (debug mode)")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda/mps)")

    args = parser.parse_args()

    # -----------------
    # Device
    # -----------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Using device: {device}")

    # -----------------
    # Step 1: Download dataset
    # -----------------
    df = download_dataset(args.repo_id_download)
    if args.sample_n or args.sample_frac:
        df = reduce_dataset(df, frac=args.sample_frac, n=args.sample_n)
        print(f"⚠️ Reduced dataset to {len(df)} rows for testing")
    print(f"✅ Full dataset size: {len(df)} rows")

    # -----------------
    # Step 2: Load model/tokenizer
    # -----------------
    if device == "cuda":
        model = AutoModel.from_pretrained("yiyanghkust/finbert-pretrain").to(device)
        use_amp = True
    else:
        model = AutoModel.from_pretrained("yiyanghkust/finbert-pretrain").to(device)
        use_amp = False
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain")



    # -----------------
    # Step 3: Encode dataset
    # -----------------
    embeddings, labels = prepare_dataset(
        df=df,
        tokenizer=tokenizer,
        model=model,
        sub_batch_size=args.sub_batch_size,
        device=device,
        n_chunks=args.n_chunks,
        batch_size=1
    )
    print("✅ Embeddings shape:", embeddings.shape)
    print("✅ Labels shape:", labels.shape)

    # -----------------
    # Step 4: Save shards
    # -----------------
    n_shards = save_sharded_dataset(
        embeddings, labels,
        max_file_size_gb=args.max_file_size_gb
    )
    print(f"✅ Dataset saved into {n_shards} shards")

    # -----------------
    # Step 5: Upload shards
    # -----------------
    upload_dataset_shards_to_hf(args.repo_id_upload, n_shards)
    print("🚀 Upload complete")
