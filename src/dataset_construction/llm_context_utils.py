# llm_context_utils.py
import os
from typing import Dict, List, Any
try:
    from litellm import completion
except ImportError:
    print("Warning: litellm library not found. Please install: pip install litellm")
    completion = None

def format_dialogue_transcript(nodes: List[Dict[str, Any]]) -> str:
    """Formats the list of dialogue nodes into a readable transcript."""
    history = []
    for node in nodes:
        speaker = node.get('speaker', 'Unknown')
        # Prioritize 'text', fallback to 'context' if text is empty
        text = node.get('text', '') or node.get('context', '')
        context = node.get('context', '')
        
        # Clean up potential HTML tags
        text = text.replace('<br>', ' ').strip()

        dialogue_line = f"{speaker}: \"{text}\""
        
        # Add context/tone if it provides extra info and is different from the main text
        if context and context.strip() and context.strip() != text.strip():
            # Clean up the "NodeContext: " prefix if present (as seen in the example JSON)
            clean_context = context.replace("NodeContext:", "").strip()
            dialogue_line += f" [Action/Tone: {clean_context}]"
            
        history.append(dialogue_line)
    return "\n".join(history)

def generate_llm_context(playthrough_segments: List[Dict[str, Any]], model_name: str) -> str:
    """
    Queries an LLM (using LiteLLM) to generate narrative context for a dialogue sequence.
    """
    if completion is None:
        return "LLM Unavailable (LiteLLM not installed)."

    # 1. Build the input from the segments
    input_details = []
    
    for i, segment in enumerate(playthrough_segments):
        synopsis = segment.get("synopsis", "N/A")
        nodes = segment.get("nodes", [])
        transcript = format_dialogue_transcript(nodes)
        
        input_details.append(f"--- Segment {i+1} ---")
        input_details.append(f"Background Synopsis: {synopsis}")
        input_details.append("Transcript:")
        input_details.append(transcript)
        input_details.append("\n")

    full_input = "\n".join(input_details)

    # 2. Define the Prompts
    system_prompt = """
    You are an expert analyst of Baldur's Gate 3 narrative and dialogue. 
    Your task is to synthesize the provided dialogue segments (including their synopses and transcripts) into a concise, cohesive narrative context.
    
    This context summary must explain the situation leading up to the very end of the provided sequence. Focus on:
    1. The overall situation and characters involved.
    2. The sequence of events and key choices made during the interaction(s).
    3. The motivations and emotional states indicated by the [Action/Tone] notes.

    Provide the summary in the third person.
    """
    
    user_prompt = f"""
    Please analyze the following dialogue sequence and provide the narrative context.

    {full_input}

    Context Summary:
    """

    # 3. Call the LLM using LiteLLM
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # API keys (e.g., GEMINI_API_KEY, OPENAI_API_KEY) are expected to be in environment variables
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )
        
        if response and response.choices:
            return response.choices[0].message.content.strip()
        else:
            return f"Error: LLM call failed or returned empty response."

    except Exception as e:
        error_msg = f"Error during LiteLLM call: {str(e)}"
        if "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg += " (Hint: Check your API Key environment variables)"
        return error_msg