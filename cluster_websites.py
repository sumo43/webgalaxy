#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cluster_websites.py - Embedding (NV-Embed-v2), KMeans(40), 3D projection, TF-IDF labels -> clusters.json

import argparse, json, os, random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Hugging Face / Torch
import torch
from transformers import AutoTokenizer, AutoModel

# ML utils
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Optional: UMAP (better 3D layout). Falls back to PCA if not available.
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Pretty progress
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    Accepts either .jsonl (one object per line) or .json (array of objects).
    Filters out items with missing/empty 'summary' field.
    """
    ext = path.suffix.lower()
    records: List[Dict[str, Any]] = []
    if ext == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception:
                    continue
    elif ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("items", [])
            records = list(data)
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .json or .jsonl")

    # Filter by summary
    clean = []
    for r in records:
        s = r.get("summary", None)
        if s is None:
            continue
        if isinstance(s, str) and s.strip():
            clean.append(r)
    return clean


def get_device_and_dtype(prefer_bfloat16=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if prefer_bfloat16 and torch.cuda.is_bf16_supported():
            return device, torch.bfloat16
        else:
            return device, torch.float16
    else:
        return torch.device("cpu"), torch.float32


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(
    texts: List[str],
    model_name: str = "nvidia/NV-Embed-v2",
    batch_size: int = 32,
    max_length: int = 512
) -> np.ndarray:
    device, dtype = get_device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True,  # or "sdpa" if FA2 not installed
    device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]

        
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}

        # Some embedding models expose pooler_output; otherwise mean-pool
        #out = model(**tok)
        out = model.encode(batch, instruction="", max_length=512)
        pooled = getattr(out, "pooler_output", None)
        #print(out)
        print(out.shape)
        embs.append(out.detach().float().cpu().numpy())
        #embs.append(out['sentence_embeddings'].float().cpu().numpy())
        #if pooled is None:
        #    pooled = mean_pool(out.last_hidden_state, tok["attention_mask"])
        #embs.append(pooled.detach().float().cpu().numpy())

    return np.concatenate(embs, axis=0)


def kmeans_cluster(embeddings: np.ndarray, k: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    return labels, centroids


def project_3d(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=seed)
        return reducer.fit_transform(embeddings)
    pca = PCA(n_components=3, random_state=seed)
    return pca.fit_transform(embeddings)


def label_clusters_tfidf(texts: List[str], labels: np.ndarray, k: int, topn: int = 3) -> List[str]:
    """
    Build simple 3-keyword labels using TF-IDF (unigram+bigram).
    For each cluster, pick the top tf-idf features.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, stop_words="english")
    X = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())

    labels_out: List[str] = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            labels_out.append("misc")
            continue
        sub = X[idx].sum(axis=0).A1
        top_idx = np.argsort(sub)[::-1][:topn]
        keys = [vocab[i] for i in top_idx]
        label = ", ".join(keys)
        labels_out.append(label if label.strip() else "misc")
    return labels_out


def build_clusters_json(
    records: List[Dict[str, Any]],
    texts: List[str],
    ids: List[str],
    titles: List[str],
    summaries: List[str],
    labels: np.ndarray,
    xyz: np.ndarray,
    cluster_keywords: List[str]
) -> Dict[str, Any]:
    k = len(set(labels))
    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            center = [0.0, 0.0, 0.0]
        else:
            center = xyz[idx].mean(axis=0).tolist()
        clusters.append({
            "id": int(c),
            "keywords": cluster_keywords[c],
            "size": int(len(idx)),
            "centroid": [float(center[0]), float(center[1]), float(center[2])]
        })

    points = []
    for i in range(len(ids)):
        points.append({
            "id": ids[i],
            "cluster": int(labels[i]),
            "pos": [float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2])],
            "title": titles[i],
            "summary": summaries[i]
        })

    return {"clusters": clusters, "points": points}


def main():
    ap = argparse.ArgumentParser(description="Cluster website summaries into K clusters and export clusters.json")
    ap.add_argument("--input", required=True, help="Path to .json or .jsonl with fields including 'summary', 'domain', 'title'")
    ap.add_argument("--output_dir", default="out_clusters", help="Folder to write outputs (clusters.json, embeddings.npy, etc.)")
    ap.add_argument("--k", type=int, default=40, help="Number of clusters (default 40)")
    ap.add_argument("--batch_size", type=int, default=8, help="Embedding batch size")
    ap.add_argument("--max_length", type=int, default=512, help="Tokenizer max length")
    ap.add_argument("--model_name", default="nvidia/NV-Embed-v2", help="Embedding model (HF name)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--label_strategy", choices=["tfidf"], default="tfidf", help="How to label clusters (default: tfidf 3-keywords)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(inp)
    if not records:
        raise SystemExit("No records with non-empty 'summary' found.")

    # Collect fields
    ids, titles, summaries = [], [], []
    for r in records:
        ids.append(str(r.get("domain") or r.get("name") or r.get("picked_url") or len(ids)))
        titles.append(str(r.get("title") or r.get("name") or ""))
        summaries.append(str(r.get("summary") or ""))

    # Embed summaries
    embeddings = embed_texts(summaries, model_name=args.model_name, batch_size=args.batch_size, max_length=args.max_length)
    np.save(out / "embeddings.npy", embeddings)

    # KMeans
    labels, centroids = kmeans_cluster(embeddings, k=args.k, seed=args.seed)
    np.save(out / "kmeans_labels.npy", labels)
    np.save(out / "kmeans_centroids.npy", centroids)

    # 3D projection
    xyz = project_3d(embeddings, seed=args.seed)
    np.save(out / "xyz.npy", xyz)

    # Cluster labels (3 keywords)
    if args.label_strategy == "tfidf":
        cluster_keywords = label_clusters_tfidf(summaries, labels, k=args.k, topn=3)
    else:
        cluster_keywords = [f"cluster-{i}" for i in range(args.k)]

    # Export a CSV summary (size + keywords)
    try:
        import csv
        from collections import Counter
        counts = Counter(labels.tolist())
        with (out / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cluster_id", "size", "keywords"])
            for i in range(args.k):
                w.writerow([i, counts.get(i, 0), cluster_keywords[i]])
    except Exception:
        pass

    # Export clusters.json for the front-end
    payload = build_clusters_json(
        records=records,
        texts=summaries,
        ids=ids, titles=titles, summaries=summaries,
        labels=labels, xyz=xyz,
        cluster_keywords=cluster_keywords
    )
    with (out / "clusters.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"), indent=2)

    print(f"Wrote: {out/'clusters.json'}")
    print("Done.")


if __name__ == "__main__":
    main()

