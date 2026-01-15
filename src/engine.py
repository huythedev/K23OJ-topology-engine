import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

class Engine:
    def __init__(self, persist_directory="data"):
        # Load embedding model
        # Switch to lightweight model for now, upgrade later.
        # 'intfloat/multilingual-e5-small' is very fast and supports Vietnamese.
        
        device = "cpu"
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        
        print(f"Initializing Engine...")
        if use_gpu:
            if torch.cuda.is_available():
                device = "cuda"
                print("GPU (CUDA) detected and enabled.")
            else:
                try:
                    import torch_directml
                    device = torch_directml.device()
                    print("GPU (DirectML) detected and enabled (AMD compatible).")
                except ImportError:
                    print("Warning: USE_GPU=true but no CUDA detected and torch-directml not installed. Falling back to CPU.")
                    print("To enable AMD support: pip install torch-directml")
                    device = "cpu"
        else:
            print("Forcing CPU execution.")
            # Set CPU threads via .env if specified
            MAX_WORKERS = os.getenv("MAX_WORKERS")
            if MAX_WORKERS:
                try:
                    num_threads = int(MAX_WORKERS)
                    torch.set_num_threads(num_threads)
                    print(f"Setting PyTorch CPU threads to {num_threads} (from .env).")
                except ValueError:
                    print(f"Invalid MAX_WORKERS in .env: {MAX_WORKERS}")

        # Load specialized CP embedding model
        print("Loading embedding model (coldchair16/CPRetriever-Prob)...")
        # Use float16 to reduce VRAM usage (2B model ~4GB in fp16 vs ~8GB in fp32)
        # This prevents OOM on 8GB cards like RX 580
        try:
            self.model = SentenceTransformer(
                'coldchair16/CPRetriever-Prob', 
                device=device, 
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16}
            )
            print("Model loaded successfully in float16.")
        except Exception as e:
            print(f"Error loading model in float16: {e}")
            print("Falling back to default precision (might OOM on 8GB VRAM)...")
            self.model = SentenceTransformer(
                'coldchair16/CPRetriever-Prob', 
                device=device, 
                trust_remote_code=True
            )
        
        self.persist_directory = persist_directory
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        # Logic extraction using LLM is disabled to save resources
        print("LLM logic extraction disabled (using raw text mode).")
        self.llm_model = None
        self.tokenizer = None
            
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

    def _extract_logic(self, raw_text: str) -> str:
        """
        Uses specialized CP model to extract core logic and remove story context.
        """
        if not self.llm_model or not self.tokenizer:
            return raw_text

        try:
            # Use chat template if available, otherwise fallback to simple prompting
            messages = [
                {"role": "user", "content": f"Extract the core algorithmic logic, constraints, and input/output format from this problem. Remove all story and flavor text.\n\nProblem: {raw_text}"}
            ]
            
            # Prepare inputs
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer(
                [text], 
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(self.llm_model.device)

            # Generate
            generated_ids = self.llm_model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=1024,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode only the new tokens
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response.strip()

        except Exception as e:
            print(f"Error during logic extraction: {e}")
            # Fallback simple generation if chat template fails
            return raw_text

    def add_problems(self, contents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """
        Embeds and stores problems.
        """
        if not contents:
            return

        # Skip explicit logic extraction since LLM is disabled
        # print(f"Extracting logic for {len(contents)} problems (this may take a while per problem)...")
        # cleaned_contents = []
        # for content in tqdm(contents, desc="Extracting Logic", unit="problem"):
        #     cleaned_contents.append(self._extract_logic(content))
        cleaned_contents = contents

        print(f"Embedding {len(cleaned_contents)} extracted problems (CPRetriever-Prob)...")
        # Direct encoding without prefix for CPRetriever
        new_embeddings = self.model.encode(cleaned_contents)
        
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
            
        # No prefix for CPRetriever
        query_embedding = self.model.encode(query_text)
        
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
