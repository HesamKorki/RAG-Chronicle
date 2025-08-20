"""Prompt templates for RAG generation."""

from typing import List


def create_system_prompt(has_passages: bool = True) -> str:
    """Create the system prompt for RAG generation."""
    if has_passages:
        return """You are a helpful assistant. Answer questions using the provided passages. Be extremely concise - aim for 1-5 words when possible. 
If the answer isn't in the passages, give a brief response using your knowledge.
Prioritize accuracy and brevity."""
    else:
        return """You are a helpful assistant. Answer questions accurately and concisely. Keep responses very brief - aim for 1-10 words when possible.
Focus on the core answer without lengthy explanations."""


def create_user_prompt(question: str, passages: List[str], doc_ids: List[int] = None) -> str:
    """Create the user prompt with question and optional retrieved passages."""
    if not passages:
        # No passages - direct question answering
        return f"Question: {question}\n\nGive a brief, direct answer:"
    
    # With passages - RAG mode
    prompt = f"Question: {question}\n\nRelevant passages:\n"
    
    for i, passage in enumerate(passages):
        if doc_ids and i < len(doc_ids):
            prompt += f"• [Doc {doc_ids[i]}]: {passage}\n"
        else:
            prompt += f"• {passage}\n"
    
    prompt += "\nBased on the passages, give a concise answer (1-5 words if possible):"
    
    return prompt


def create_chat_messages(question: str, passages: List[str], doc_ids: List[int] = None) -> List[dict]:
    """Create chat messages for the Qwen model."""
    has_passages = bool(passages)
    system_prompt = create_system_prompt(has_passages)
    user_prompt = create_user_prompt(question, passages, doc_ids)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def truncate_passages_for_context(passages: List[str], doc_ids: List[int], 
                                max_tokens: int = 2048, avg_tokens_per_char: float = 0.25) -> tuple:
    """Truncate passages to fit within context window."""
    if not passages:
        return [], []
    
    # Estimate tokens for each passage
    passage_tokens = []
    for passage in passages:
        estimated_tokens = len(passage) * avg_tokens_per_char
        passage_tokens.append(estimated_tokens)
    
    # Reserve tokens for question and formatting
    reserved_tokens = 200  # Approximate tokens for question and formatting
    available_tokens = max_tokens - reserved_tokens
    
    # Select passages that fit
    selected_passages = []
    selected_doc_ids = []
    current_tokens = 0
    
    for i, (passage, doc_id) in enumerate(zip(passages, doc_ids or [])):
        passage_token_count = passage_tokens[i]
        
        if current_tokens + passage_token_count <= available_tokens:
            selected_passages.append(passage)
            selected_doc_ids.append(doc_id)
            current_tokens += passage_token_count
        else:
            break
    
    return selected_passages, selected_doc_ids


def format_passages_for_display(passages: List[str], doc_ids: List[int] = None, 
                               max_length: int = 500) -> str:
    """Format passages for display purposes."""
    if not passages:
        return "No passages provided."
    
    formatted = []
    for i, passage in enumerate(passages):
        # Truncate long passages
        if len(passage) > max_length:
            passage = passage[:max_length] + "..."
        
        if doc_ids and i < len(doc_ids):
            formatted.append(f"[Doc {doc_ids[i]}]: {passage}")
        else:
            formatted.append(passage)
    
    return "\n\n".join(formatted)


def create_debug_prompt(question: str, passages: List[str], doc_ids: List[int] = None,
                       retrieved_scores: List[float] = None) -> str:
    """Create a debug prompt with additional information."""
    prompt = f"Question: {question}\n\n"
    
    if retrieved_scores:
        prompt += "Retrieved passages with scores:\n"
        for i, (passage, score) in enumerate(zip(passages, retrieved_scores)):
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else i
            prompt += f"• [Doc {doc_id}, Score: {score:.4f}]: {passage[:200]}...\n"
    else:
        prompt += "Retrieved passages:\n"
        for i, passage in enumerate(passages):
            doc_id = doc_ids[i] if doc_ids and i < len(doc_ids) else i
            prompt += f"• [Doc {doc_id}]: {passage[:200]}...\n"
    
    prompt += "\nAnswer based on the passages above:"
    
    return prompt
