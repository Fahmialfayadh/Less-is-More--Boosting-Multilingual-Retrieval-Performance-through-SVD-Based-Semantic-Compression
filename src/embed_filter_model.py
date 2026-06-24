import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class EmbedFilterPipeline:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", v_noise_path: str = None):
        """
        Initializes the model, tokenizer, and loads the Edge Spectrum matrix (v_noise).
        """
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Using AutoModel for generic hidden state extraction (instead of ForMaskedLM)
        # Actually, standard AutoModel is safer for generic feature extraction
        from transformers import AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.v_noise = None
        if v_noise_path:
            print(f"Loading Edge Spectrum (v_noise) from: {v_noise_path}")
            self.v_noise = torch.load(v_noise_path, map_location="cpu", weights_only=True)
            assert self.v_noise is not None, "Failed to load v_noise."
            print(f"v_noise shape: {self.v_noise.shape}")
            
    def get_raw_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Generates raw embeddings for a list of texts by mean pooling over token embeddings.
        """
        # E5 models heavily rely on the "query: " prefix for optimal accuracy
        if "e5" in self.model.name_or_path.lower():
            texts = [f"query: {t}" for t in texts]
            
        # Process in batches to avoid OOM
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state (dim 768)
                hidden_states = outputs.last_hidden_state
                
                # Mean pooling (excluding padding tokens)
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                
                all_embeddings.append(mean_pooled.cpu())
                
        return torch.cat(all_embeddings, dim=0)
        
    def encode_baseline(self, texts: list[str]) -> np.ndarray:
        """Returns raw embeddings as NumPy arrays."""
        return self.get_raw_embeddings(texts).numpy()
        
    def encode_filtered(self, texts: list[str] = None, k: int = 10, tau: int = None, precomputed_raw_embs: torch.Tensor = None) -> np.ndarray:
        """
        Returns filtered embeddings applied with the projection matrix post-hoc.
        - k: number of Edge Spectrum (noise) components to remove.
        - tau: compression size (Bulk Spectrum dimensions to keep). If None, no compression.
        - precomputed_raw_embs: Optional tensor of raw embeddings to speed up loops.
        """
        if precomputed_raw_embs is not None:
            raw_embs = precomputed_raw_embs
        else:
            raw_embs = self.get_raw_embeddings(texts)
            
        if self.v_noise is not None:
            # Select top k noise components
            v_k = self.v_noise[:, :k] # Shape: (Hidden, k)
            
            # Create projection matrix: P = I - v_k * v_k^T
            hidden_dim = v_k.shape[0]
            I = torch.eye(hidden_dim, device=raw_embs.device)
            P = I - torch.mm(v_k, v_k.t())
            
            # Apply Noise Filter
            filtered_embs = torch.mm(raw_embs, P)
            
            # Compression (tau) via Truncated SVD on the Bulk Spectrum
            if tau is not None and tau < hidden_dim:
                # Center the filtered embeddings
                mean_emb = filtered_embs.mean(dim=0, keepdim=True)
                centered_emb = filtered_embs - mean_emb
                
                # Perform PCA to get the top tau components of the Bulk Spectrum
                q_val = min(tau, centered_emb.size(0), centered_emb.size(1))
                U, S, V = torch.pca_lowrank(centered_emb, q=q_val, center=False, niter=3)
                
                # Project onto the Bulk Spectrum (compression)
                # E_compressed = (E_filtered - mean) * V
                compressed_embs = torch.mm(centered_emb, V)
                return compressed_embs.numpy()
                
            return filtered_embs.numpy()
        else:
            print("Warning: No v_noise matrix loaded. Returning baseline embeddings.")
            return raw_embs.numpy()
