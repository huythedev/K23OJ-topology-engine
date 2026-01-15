import os
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Engine:
    def __init__(self, persist_directory="data"):
        # Load embedding model
        # Switch to lightweight model for now, upgrade later.
        # 'intfloat/multilingual-e5-small' is very fast and supports Vietnamese.
        
        device = "cpu"
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        
        print(f"Loading embedding model (intfloat/multilingual-e5-small)...")
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
                print("GPU (CUDA) detected and enabled.")
            else:
                try:
                    import torch_directml
                    device = torch_directml.device()
                    print("GPU (DirectML) detected and enabled.")
                except ImportError:
                    print("Warning: USE_GPU=true but no CUDA detected and torch-directml not installed. Falling back to CPU.")
                    device = "cpu"
        else:
            print("Forcing CPU execution.")

        self.model = SentenceTransformer('intfloat/multilingual-e5-small', device=device)
        
        self.persist_directory = persist_directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        self.index_path = os.path.join(self.persist_directory, "index.pkl")
        self.data = {
            "embeddings": [],
            "metadatas": [],
            "ids": []
        }
        self.load_index()

        # Check if loaded index dimensions match the model
        if len(self.data["embeddings"]) > 0:
            index_dim = self.data["embeddings"].shape[1]
            model_dim = self.model.get_sentence_embedding_dimension()
            if index_dim != model_dim:
                print(f"Dimension mismatch! Index: {index_dim}, Model: {model_dim}.")
                print("Clearing incompatible index. Please re-run indexing.")
                self.reset_index()

    def load_index(self):
        if os.path.exists(self.index_path):
            print(f"Loading index from {self.index_path}...")
            try:
                with open(self.index_path, 'rb') as f:
                    self.data = pickle.load(f)
            except Exception as e:
                print(f"Error loading index: {e}")

    def reset_index(self):
        """
        Clears the in-memory index and deletes the on-disk file.
        """
        self.data = {
            "embeddings": [],
            "metadatas": [],
            "ids": []
        }
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                print("Existing index cleared.")
            except Exception as e:
                print(f"Error clearing index: {e}")

    def save_index(self):
        print(f"Saving index to {self.index_path}...")
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.data, f)

    def add_problems(self, contents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Embeds and stores problems.
        """
        if not contents:
            return

        print(f"Embedding {len(contents)} problems (e5-small)...")
        # Add "passage: " prefix for E5 models
        prefixed_contents = ["passage: " + c for c in contents]
        new_embeddings = self.model.encode(prefixed_contents) # Returns numpy array
        
        # Determine current size to handle appending correctly
        if len(self.data["embeddings"]) == 0:
            self.data["embeddings"] = new_embeddings
        else:
            self.data["embeddings"] = np.concatenate((self.data["embeddings"], new_embeddings), axis=0)
            
        self.data["metadatas"].extend(metadatas)
        self.data["ids"].extend(ids)
        
        # self.save_index() # Don't save on every batch, too slow! User must call explicitly.

    def search(self, query_text: str, n_results: int = 5):
        """
        Searches for similar problems using cosine similarity.
        """
        if len(self.data["embeddings"]) == 0:
            return None
            
        # Add "query: " prefix for E5 models
        query_embedding = self.model.encode("query: " + query_text)
        
        # Calculate cosine similarity
        # util.cos_sim returns tensor
        corpus_embeddings = self.data["embeddings"]
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get top k results
        top_k_indices = np.argsort(-scores.numpy())[:n_results]
        
        found_ids = []
        found_dists = [] # We have scores (similarity), not distances. Distance = 1 - Similarity roughly.
        found_metas = []
        
        for idx in top_k_indices:
            score = scores[idx].item()
            found_ids.append(self.data["ids"][idx])
            found_dists.append(1 - score) # Convert similarity to "distance" for compatibility
            found_metas.append(self.data["metadatas"][idx])
            
        return {
            'ids': [found_ids],
            'distances': [found_dists],
            'metadatas': [found_metas]
        }
