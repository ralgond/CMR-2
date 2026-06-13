"""
embedder.py
Local Qwen3-Embedding-0.6B wrapper for query-side encoding.

The track embeddings in Track-Embedding/ were produced by this same model
(column: metadata-qwen3_embedding_0.6b, dim=1024), so query vectors and
track vectors are in the same space — direct cosine similarity works.

Model path: /root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = os.environ.get(
    "QWEN3_EMBED_PATH",
    "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"
)

# Qwen3-Embedding task prefix for retrieval queries
# https://huggingface.co/Qwen/Qwen3-Embedding
QUERY_TASK = "Given a music preference description, retrieve relevant tracks"

_tokenizer = None
_model = None
_device = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return

    print(f"[embedder] Loading Qwen3-Embedding from {MODEL_PATH} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(_device).eval()
    print(f"[embedder] Model loaded on {_device}")


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Qwen3-Embedding uses last-token pooling."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        seq_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lengths]


def _format_query(text: str, task: str = QUERY_TASK) -> str:
    """Qwen3-Embedding query format: <instruct>task\n<query>text."""
    return f"<instruct>{task}\n<query>{text}"


def encode_queries(texts: list[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
    """
    Encode a list of query strings → L2-normalized float32 numpy array (N, 1024).
    Uses task-instruction prefix for query-side encoding.
    """
    _load_model()
    formatted = [_format_query(t) for t in texts]
    all_embeddings = []

    for i in range(0, len(formatted), batch_size):
        batch = formatted[i: i + batch_size]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**encoded)

        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def encode_single(text: str) -> np.ndarray:
    """Convenience wrapper for a single query string → (1024,) numpy array."""
    return encode_queries([text])[0]


def encode_documents(texts: list[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
    """
    Encode document/track texts (no task prefix) → L2-normalized (N, 1024).
    Used if you need to re-embed track metadata on the fly.
    NOTE: precomputed embeddings from Track-Embedding/ are preferred.
    """
    _load_model()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            outputs = _model(**encoded)

        embeddings = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ──────────────────────────────────────────────
# SigLIP2 text encoder — for album cover queries
# Model: google/siglip2-base-patch16-224 (dim=768)
# Same model used to produce image-siglip2 track embeddings,
# so text query vec and cover vec are in the same space.
# ──────────────────────────────────────────────

SIGLIP2_MODEL_PATH = os.environ.get(
    "SIGLIP2_MODEL_PATH",
    "/root/.cache/modelscope/hub/models/google/siglip2-base-patch16-224",
)

_siglip_tokenizer = None
_siglip_model     = None
_siglip_device    = None


def _load_siglip():
    global _siglip_tokenizer, _siglip_model, _siglip_device
    if _siglip_model is not None:
        return
    print(f"[embedder] Loading SigLIP2 from {SIGLIP2_MODEL_PATH} ...")
    from transformers import AutoProcessor, AutoModel
    _siglip_tokenizer = AutoProcessor.from_pretrained(
        SIGLIP2_MODEL_PATH, trust_remote_code=True
    )
    _siglip_model = AutoModel.from_pretrained(
        SIGLIP2_MODEL_PATH, trust_remote_code=True
    )
    _siglip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _siglip_model = _siglip_model.to(_siglip_device).eval()
    print(f"[embedder] SigLIP2 loaded on {_siglip_device}")


def encode_cover_query(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """
    Encode cover description texts via SigLIP2 text encoder.
    Returns L2-normalized (N, 768) float32 array.
    Same embedding space as image-siglip2 column in Track-Embedding/.
    """
    _load_siglip()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = _siglip_tokenizer(
            text=batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(_siglip_device)

        with torch.no_grad():
            outputs = _siglip_model.get_text_features(**inputs)

        # get_text_features may return a tensor or BaseModelOutputWithPooling
        # depending on the model version — handle both
        if isinstance(outputs, torch.Tensor):
            text_embeds = outputs
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_embeds = outputs.pooler_output
        else:
            # fallback: CLS token from last hidden state
            text_embeds = outputs.last_hidden_state[:, 0, :]

        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        all_embeddings.append(text_embeds.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)