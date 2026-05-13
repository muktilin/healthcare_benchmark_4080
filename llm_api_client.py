#!/usr/bin/env python3
"""
LLM API Client - Send prompts to Gradio-based FPGA LLM server
Compatible with FPGA_DS_1024.py Gradio interface
Uses direct HTTP calls to avoid gradio_client library issues
"""

import requests
import json
import sys
from typing import Optional, Dict, Any, List


class LLMClient:
    """Client for interacting with Gradio LLM server (FPGA_DS_1024.py)"""
    
    def __init__(
        self, 
        server_url: str = "http://192.168.115.190:8080",
        model: str = "nvidia/DLER-R1-1.5B-Research",
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        no_think: bool = False,
        concise_prefix: bool = True,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 2.0,
        repetition_penalty: float = 4.0,
        repetition_penalty_window: int = 6,
        timeout: int = 120
    ):
        """
        Initialize Gradio LLM API client
        
        Args:
            server_url: Gradio server base URL (e.g., http://localhost:8080)
            model: Model name (e.g., nvidia/DLER-R1-1.5B-Research)
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Top-p nucleus sampling (0.0 to 1.0)
            top_k: Top-k sampling (0 to 200)
            no_think: Disable thinking/reasoning mode
            concise_prefix: Prepend concise instruction to prompts
            presence_penalty: Penalty for token presence (0.0 to 2.0)
            frequency_penalty: Penalty for token frequency (0.0 to 2.0)
            repetition_penalty: Penalty for repetition (0.0 to 5.0)
            repetition_penalty_window: Window size for repetition penalty
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.no_think = no_think
        self.concise_prefix = concise_prefix
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.repetition_penalty_window = repetition_penalty_window
        self.timeout = timeout
        self.conversation_history: List[Dict[str, str]] = []
        self.session = requests.Session()  # Reuse connection
        self.api_prefix = "/gradio_api"
        
    def send_prompt(
        self, 
        prompt: str,
        reset_conversation: bool = False
    ) -> str:
        """
        Send a prompt to the Gradio LLM server
        
        Args:
            prompt: User prompt/question
            reset_conversation: Whether to reset conversation history
            
        Returns:
            LLM response text
        """
        if reset_conversation:
            self.conversation_history = []
        
        # Gradio API endpoint structure (newer Gradio versions)
        # We need to find the correct function API endpoint
        # Try multiple endpoint patterns
        
        endpoints_to_try = [
            f"{self.api_prefix}/call/process_text_input",  # Gradio 5+
            "/call/process_text_input",  # Legacy/no prefix fallback
            f"{self.api_prefix}/api/process_text_input",
            f"{self.api_prefix}/run/process_text_input",
            "/api/process_text_input",
            "/run/process_text_input",
            "/api/predict",  # Older format
        ]
        
        # Prepare function arguments
        # The process_text_input function signature:
        # text_input, conversation, model_name, no_think, concise_prefix, 
        # temperature, top_p, top_k, presence_penalty, frequency_penalty,
        # repetition_penalty, repetition_penalty_window
        
        data_payload = [
            prompt,  # text_input
            self.conversation_history,  # conversation
            self.model,  # model_name
            self.no_think,  # no_think
            self.concise_prefix,  # concise_prefix
            self.temperature,  # temperature
            self.top_p,  # top_p
            self.top_k,  # top_k
            self.presence_penalty,  # presence_penalty
            self.frequency_penalty,  # frequency_penalty
            self.repetition_penalty,  # repetition_penalty
            self.repetition_penalty_window  # repetition_penalty_window
        ]
        
        # Try modern Gradio API first (streaming with event_id)
        try:
            return self._call_gradio_streaming(data_payload)
        except Exception as e:
            print(f"[DEBUG] Streaming API failed: {e}")
            # Fallback to direct API calls
            pass
        
        # Try different endpoint patterns
        for endpoint in endpoints_to_try:
            try:
                api_url = f"{self.server_url}{endpoint}"
                payload = {"data": data_payload}
                
                response = requests.post(
                    api_url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return self._extract_response_from_result(result)
                    
            except Exception as e:
                print(f"[DEBUG] Tried {endpoint}: {e}")
                continue
        
        return f"Error: Could not connect to Gradio API. Please ensure the server is running at {self.server_url}"
    
    def _call_gradio_streaming(self, data_payload: List) -> str:
        """
        Call Gradio using modern streaming API
        
        Args:
            data_payload: Function arguments
            
        Returns:
            Response text
        """
        # Step 1: Initiate the call
        call_endpoint = f"{self.server_url}{self.api_prefix}/call/process_text_input"
        
        response = requests.post(
            call_endpoint,
            json={"data": data_payload},
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to initiate call: {response.status_code}")
        
        result = response.json()
        event_id = result.get("event_id")
        
        if not event_id:
            raise Exception("No event_id returned from call")
        
        # Step 2: Read the SSE stream once until completion.
        stream_endpoint = f"{self.server_url}{self.api_prefix}/call/process_text_input/{event_id}"
        with self.session.get(stream_endpoint, stream=True, timeout=self.timeout) as stream_resp:
            if stream_resp.status_code != 200:
                raise Exception(f"Failed to open stream: {stream_resp.status_code}")

            current_event = ""
            last_conversation = None
            for raw_line in stream_resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue

                if raw_line.startswith("event: "):
                    current_event = raw_line[7:].strip().lower()
                    continue

                if not raw_line.startswith("data: "):
                    continue

                data_json = raw_line[6:]
                try:
                    event_data = json.loads(data_json)
                except json.JSONDecodeError:
                    continue

                # For this Gradio server, payload is usually:
                # [updated_conversation, ""]
                if isinstance(event_data, list) and len(event_data) > 0:
                    last_conversation = event_data[0]

                # Some Gradio versions may send a dict completion event.
                if isinstance(event_data, dict) and event_data.get("msg") == "process_completed":
                    output_data = event_data.get("output", {}).get("data", [])
                    if output_data and len(output_data) > 0:
                        last_conversation = output_data[0]

                if current_event != "complete":
                    continue

                if not last_conversation:
                    return "No response from model"

                self.conversation_history = last_conversation

                for msg in reversed(last_conversation):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return self._clean_response(msg.get("content", ""))

                return "No response from model"

        raise Exception("No completion event received from stream")
    
    def _extract_response_from_result(self, result: Dict) -> str:
        """
        Extract response from various Gradio result formats
        
        Args:
            result: API response
            
        Returns:
            Extracted response text
        """
        # Gradio returns data in "data" field
        # The output is [conversation, text_input]
        if "data" in result and len(result["data"]) > 0:
            updated_conversation = result["data"][0]
            
            # Update conversation history
            self.conversation_history = updated_conversation
            
            # Extract last assistant message
            for msg in reversed(updated_conversation):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return self._clean_response(msg.get("content", ""))
            
            return "No response from model"
        
        return f"Unexpected response format: {result}"
    
    def _clean_response(self, content: str) -> str:
        """
        Clean response content by removing HTML formatting
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned text
        """
        # Remove details/summary tags for thinking
        import re
        
        # Remove thinking details blocks
        content = re.sub(r'<details[^>]*>.*?</details>', '', content, flags=re.DOTALL)
        
        # Remove horizontal rules
        content = content.replace('---', '').strip()
        
        return content.strip()
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history


def main():
    """Main function for command-line usage"""
    
    # Configuration for FPGA_DS_1024.py Gradio server
    # SERVER_URL = "http://localhost:8080"
    SERVER_URL = "http://192.168.115.190:8080"
    MODEL = "nvidia/DLER-R1-1.5B-Research"  # Available models:
    # - nvidia/DLER-R1-1.5B-Research
    # - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # - Qwen/Qwen3-1.7B
    # - meta-llama/Llama-3.2-1B-Instruct
    
    # Initialize client
    client = LLMClient(
        server_url=SERVER_URL,
        model=MODEL,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        no_think=False,  # Enable thinking mode
        concise_prefix=True,
        presence_penalty=0.0,
        frequency_penalty=2.0,
        repetition_penalty=4.0,
        repetition_penalty_window=6
    )
    
    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        # Default example prompt
        prompt = "What is the capital of France?"
    
    print(f"Sending prompt to {SERVER_URL}")
    print(f"Model: {MODEL}")
    print(f"Prompt: {prompt}\n")
    print("=" * 60)
    
    # Send request and get response
    response = client.send_prompt(prompt, reset_conversation=True)
    
    # Print response
    print("LLM Response:")
    print("-" * 60)
    print(response)
    print("=" * 60)
    
    # Example: Multi-turn conversation
    if len(sys.argv) == 1:  # Only in demo mode
        print("\n\n=== Multi-turn Conversation Example ===\n")
        
        # Don't reset - continue conversation
        follow_up = "What is the population of that city?"
        print(f"Follow-up: {follow_up}\n")
        print("=" * 60)
        
        response2 = client.send_prompt(follow_up, reset_conversation=False)
        
        print("LLM Response:")
        print("-" * 60)
        print(response2)
        print("=" * 60)


if __name__ == "__main__":
    main()
