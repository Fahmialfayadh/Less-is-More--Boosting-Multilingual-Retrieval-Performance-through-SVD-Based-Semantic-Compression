import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

class QwenEmbedFilterPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", v_noise_path_id: str = None, v_noise_path_en: str = None):
        """
        Initializes the model, tokenizer, and loads the Edge Spectrum matrices.
        """
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # We need left padding for Last-Token pooling, and left truncation to preserve the prompt suffix
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Decoder-Only Model with float16 to save VRAM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()
        
        self.v_noise_id = None
        if v_noise_path_id and os.path.exists(v_noise_path_id):
            print(f"Loading ID Edge Spectrum from: {v_noise_path_id}")
            self.v_noise_id = torch.load(v_noise_path_id, map_location="cpu", weights_only=True)
            
        self.v_noise_en = None
        if v_noise_path_en and os.path.exists(v_noise_path_en):
            print(f"Loading EN Edge Spectrum from: {v_noise_path_en}")
            self.v_noise_en = torch.load(v_noise_path_en, map_location="cpu", weights_only=True)

    def get_raw_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Generates raw embeddings for a list of texts using Last-Token pooling over token embeddings.
        """
        batch_size = 8 # Smaller batch size for 1.5B model to prevent OOM
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Apply PromptEOL format to enforce task-awareness (Summarize semantics rather than just predicting syntax)
            prompts = [f'Summarize the sentence: "{text}" in one word:"' for text in batch_texts]
            
            inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # For causal LM, we use output_hidden_states to get the representation
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # hidden_states is a tuple. The last element is the last layer's hidden states
                hidden_states = outputs.hidden_states[-1] 
                
                # Last-Token pooling (since padding_side="left", the last token is always at index -1)
                last_token_hidden_states = hidden_states[:, -1, :]
                
                all_embeddings.append(last_token_hidden_states.cpu().float()) # Convert to float32 for metric calculations
                
        return torch.cat(all_embeddings, dim=0)

    def encode_baseline(self, texts: list[str]) -> np.ndarray:
        return self.get_raw_embeddings(texts).numpy()

    def encode_filtered(self, precomputed_raw_embs: torch.Tensor, filter_type: str = "id", k: int = 10) -> np.ndarray:
        """
        Applies orthogonal projection post-hoc.
        filter_type: "id" or "en"
        k: number of noise components to remove.
        """
        raw_embs = precomputed_raw_embs
        
        if filter_type == "id":
            v_noise = self.v_noise_id
        elif filter_type == "en":
            v_noise = self.v_noise_en
        else:
            raise ValueError("filter_type must be 'id' or 'en'")
            
        if v_noise is not None:
            v_k = v_noise[:, :k] # Shape: (Hidden, k)
            hidden_dim = v_k.shape[0]
            
            # P = I - v_k * v_k^T
            I = torch.eye(hidden_dim, device=raw_embs.device)
            P = I - torch.mm(v_k, v_k.t())
            
            # Apply Filter
            filtered_embs = torch.mm(raw_embs, P)
            return filtered_embs.numpy()
        else:
            print(f"Warning: No {filter_type} v_noise matrix loaded. Returning baseline embeddings.")
    def encode_retention_window(self, precomputed_raw_embs: torch.Tensor, start_dim: int = 0, end_dim: int = 768) -> np.ndarray:
        """
        Reduces dimension from 1536 to (end_dim - start_dim) by retaining only the specified principal components.
        This implements the "Head-to-Middle Retention" window technique (e.g. 0:768).
        """
        if not hasattr(self, 'Vh'):
            import time
            print("Computing SVD to get Vh for PCA projection...")
            W = self.model.lm_head.weight.detach().float()
            start_time = time.time()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            self.Vh = Vh.to(self.device)
            print(f"SVD computed in {time.time() - start_time:.2f} seconds.")
            
        V_opt = self.Vh[start_dim:end_dim, :] # Shape: (Window_Size, 1536)
        
        # PCA Projection: raw_embs @ V_opt.T
        # raw_embs is (N, 1536), V_opt.T is (1536, Window_Size)
        # Resulting shape: (N, Window_Size)
        filtered_embs = torch.matmul(precomputed_raw_embs.to(self.device), V_opt.t())
        return filtered_embs.cpu().numpy()

