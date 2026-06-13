import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0___6B"
QUERY_TASK = "Given a music preference description, retrieve relevant tracks"


class QwenEmbedder:
    def __init__(self):
        print(f"[embedder] Loading Qwen3-Embedding from {MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        print(f"[embedder] Model loaded on {self.device}")

    def _format_query(self, text: str, task: str = QUERY_TASK) -> str:
        """Qwen3-Embedding query format: <instruct>task\n<query>text."""
        return f"<instruct>{task}\n<query>{text}"

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Qwen3-Embedding uses last-token pooling."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            seq_lengths = attention_mask.sum(dim=1) - 1
            return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), seq_lengths]
    
    def encode_queries(self, texts: list[str], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
        """
        Encode a list of query strings → L2-normalized float32 numpy array (N, 1024).
        Uses task-instruction prefix for query-side encoding.
        """
        formatted = [self._format_query(t) for t in texts]
        all_embeddings = []
    
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i: i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
    
            with torch.no_grad():
                outputs = self.model(**encoded)
    
            embeddings = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())
    
        return np.concatenate(all_embeddings, axis=0)
    
    def __call__(self, text) -> np.ndarray:
        """Convenience wrapper for a single query string → (1024,) numpy array."""
        return self.encode_queries([text])[0]


def test():
    qwen_embedder = QwenEmbedder()
    print(qwen_embedder("Find for me."))

if __name__ == "__main__":
    test()