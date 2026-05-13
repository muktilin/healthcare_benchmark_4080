from tool.llm_api_client import LLMClient

# Initialize client
client = LLMClient(
    server_url="http://192.168.115.190:8080",
    model="nvidia/DLER-R1-1.5B-Research",
    temperature=0.6,
    no_think=False  # Enable thinking mode
)

# Single query
response = client.send_prompt("What is the chemical symbol for water?")
print(response)
client.reset_conversation()

# Multi-turn conversation
response1 = client.send_prompt("What is the capital of France?")
response2 = client.send_prompt("What is its population?")  # Maintains context
print(response1)
print(response2)
# Reset conversation
client.reset_conversation()