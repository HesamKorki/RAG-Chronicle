"""Qwen generator implementation for RAG experiments."""

import torch
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from .prompts import create_chat_messages, truncate_passages_for_context
from ..config import GeneratorConfig
from ..utils.seeds import set_seed
from ..utils.timing import Timer


class QwenGenerator:
    """Qwen generator for RAG experiments."""
    
    def __init__(self, config: GeneratorConfig):
        """Initialize the Qwen generator."""
        self.config = config
        self.model_name = config.model_name
        self.device = config.device.device
        self.torch_dtype = config.device.torch_dtype
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Set seed for reproducibility
        set_seed(config.seed)
    
    def _load_model(self) -> None:
        """Load the Qwen model and tokenizer with MPS fallback."""
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Attempt to load model with error handling for MPS
        try:
            if self.device == "mps":
                print("Attempting to load on MPS with optimizations...")
                # Optimize for MPS: use float16 and lower memory usage
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,  # Use float16 for MPS to save memory
                    low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
                    device_map={"": "mps"}      # Force MPS placement
                )
                print(f"Model loaded successfully on MPS")
            elif self.device == "auto":
                # Use device_map for automatic device placement
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                print(f"Model loaded successfully with auto device placement")
            else:
                # Load to CPU first, then move to specific device
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
                print(f"Model loaded successfully on {self.device}")
                
        except Exception as e:
            if self.device == "mps":
                print(f"MPS loading failed: {e}")
                print("Falling back to CPU...")
                self.device = "cpu"
                # Load on CPU as fallback
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True
                )
                print(f"Model loaded successfully on CPU (fallback)")
            else:
                print(f"Model loading failed: {e}")
                raise
    
    def generate(self, question: str, passages: List[str], doc_ids: Optional[List[int]] = None,
                k: int = 5) -> Dict[str, Any]:
        """Generate answer using retrieved passages or general knowledge."""
        # No early return - we can answer with or without passages
        
        with Timer("Generation") as timer:
            # Clear MPS cache if using MPS
            if self.device == "mps" and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass  # Ignore cache clearing errors
            
            # Truncate passages to fit context window
            selected_passages, selected_doc_ids = truncate_passages_for_context(
                passages, doc_ids, self.config.context_token_budget
            )
            
            # Create chat messages
            messages = create_chat_messages(question, selected_passages, selected_doc_ids)
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize input with attention mask
            inputs = self.tokenizer([text], return_tensors="pt", padding=True)
            
            # Move inputs to appropriate device
            try:
                if hasattr(self.model, 'device') and hasattr(next(self.model.parameters()), 'device'):
                    # If using device_map, move inputs to model's device
                    model_device = next(self.model.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                elif self.device not in ["auto"]:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as e:
                print(f"Warning: Could not move inputs to device: {e}")
                # Keep inputs on CPU as fallback
            
            # Generate
            generation_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Only add sampling parameters if temperature > 0
            if self.config.temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                })
            else:
                generation_kwargs["do_sample"] = False
            
            # Try generation with error handling for MPS
            try:
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
            except Exception as e:
                if "MPSTemporaryNDArray" in str(e) or "total bytes of NDArray > 2**32" in str(e):
                    print(f"MPS generation failed due to memory: {e}")
                    print("Moving model to CPU for generation...")
                    
                    # Move model to CPU
                    self.model = self.model.cpu()
                    self.device = "cpu"
                    
                    # Move inputs to CPU
                    inputs = {k: v.cpu() for k, v in inputs.items()}
                    
                    # Try generation on CPU
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                    
                    print("Generation successful on CPU")
                else:
                    raise e
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            input_length = inputs["input_ids"].shape[1]
            
            if len(outputs[0]) > input_length:
                generated_part = self.tokenizer.decode(
                    outputs[0][input_length:], 
                    skip_special_tokens=True
                ).strip()
            else:
                generated_part = ""
            
            # Count tokens
            tokens_generated = len(outputs[0]) - input_length
            
            # Clear MPS cache after generation
            if self.device == "mps" and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass  # Ignore cache clearing errors
        
        return {
            "answer": generated_part,
            "passages_used": selected_passages,
            "doc_ids_used": selected_doc_ids,
            "generation_time": timer.elapsed_time,
            "tokens_generated": tokens_generated,
            "full_prompt": text,
            "full_response": generated_text
        }
    
    def generate_batch(self, questions: List[str], passages_list: List[List[str]], 
                      doc_ids_list: Optional[List[List[int]]] = None,
                      k: int = 5) -> List[Dict[str, Any]]:
        """Generate answers for a batch of questions."""
        results = []
        
        for i, question in enumerate(questions):
            passages = passages_list[i] if i < len(passages_list) else []
            doc_ids = doc_ids_list[i] if doc_ids_list and i < len(doc_ids_list) else None
            
            result = self.generate(question, passages, doc_ids, k)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "context_token_budget": self.config.context_token_budget,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        if self.tokenizer is None:
            return len(text.split())  # Rough estimate
        
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def test_generation(self, test_question: str = "What is the capital of France?") -> Dict[str, Any]:
        """Test the generation with a simple question."""
        test_passages = ["Paris is the capital and largest city of France."]
        
        result = self.generate(test_question, test_passages)
        
        return {
            "test_question": test_question,
            "test_passages": test_passages,
            "result": result,
            "model_info": self.get_model_info()
        }
