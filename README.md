# 🚀 Server Data Processing

This repository contains a pipeline to process large text datasets into **FinBERT embeddings**, shard them into manageable files, and upload them to the Hugging Face Hub for further use.

---

## 📂 Repository Structure

- `main.py` → Main script to download, preprocess, encode, shard, and upload the dataset.  
- `requirements.txt` → Dependencies required for running the pipeline.  
- `pipeline.sh` → Universal script to set up the environment and run the pipeline (works on local servers and managed environments like Hugging Face Studio).  
- `pre_dataset.parquet` → Example preprocessed dataset (can be replaced with your own).  
- `embedding_dataset_prep.ipynb` → Notebook for step-by-step dataset preparation.  
- `test_model.ipynb` → Notebook for testing models with the produced embeddings.  

---

## ⚡ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/Arthurmaffre34/server_data_processing.git
cd server_data_processing
```

### 2. Make scripts executable (important!)
If you just cloned the repo, make sure the `.sh` scripts are executable:
```bash
chmod +x pipeline.sh
```

### 3. Configure Hugging Face
Before running, export your Hugging Face token (required for private repos or large file uploads):
```bash
export HF_TOKEN="your_huggingface_token"
```

### 4. Run the pipeline
The pipeline automatically detects whether you are on a normal server or a managed environment (like Hugging Face Studio).  
- On a **server/local machine** → creates a `venv` and installs dependencies.  
- On a **managed Studio** → skips venv and installs into the default environment.  

Run with:
```bash
./pipeline.sh --repo_id_download "Arthurmaffre34/pre-dataset"               --repo_id_upload "Gill-Hack-25-UdeM/Text_Embedding_Dataset"               --sub_batch_size 16               --n_chunks 100               --batch_size 1               --max_file_size_gb 0.5
```

---

## ⚙️ Parameters

| Argument            | Description |
|---------------------|-------------|
| `--repo_id_download` | Hugging Face dataset repo to download from |
| `--repo_id_upload`   | Hugging Face dataset repo to upload shards |
| `--sub_batch_size`   | Number of text chunks processed at once inside FinBERT |
| `--n_chunks`         | Maximum chunks per document |
| `--batch_size`       | DataLoader batch size |
| `--max_file_size_gb` | Target maximum shard size (default: 1 GB) |

---

## ✅ Example Output

- Embeddings tensor: `[num_samples, 2, n_chunks, hidden_dim]`  
- Labels tensor: `[num_samples]`  
- Sharded files: `finbert_embeddings_part0.pt, finbert_embeddings_part1.pt, ...`  
  Each uploaded automatically to Hugging Face.

---

## 📌 Notes

- Ensure `max_length ≤ 512` (FinBERT limitation).  
- `sub_batch_size` cannot exceed `n_chunks`.  
- Tokenizer must match **FinBERT**, otherwise a warning is raised.  
- GPU (`cuda`) or Apple Silicon (`mps`) is automatically detected.  
- Always run `chmod +x` on `.sh` files if you have execution permission errors.  

---

## 🔗 Hugging Face Repos

- **Input dataset** → [Arthurmaffre34/pre-dataset](https://huggingface.co/datasets/Arthurmaffre34/pre-dataset)  
- **Processed embeddings** → [Gill-Hack-25-UdeM/Text_Embedding_Dataset](https://huggingface.co/datasets/Gill-Hack-25-UdeM/Text_Embedding_Dataset)
