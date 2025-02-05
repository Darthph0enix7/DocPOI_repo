from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type
import json
import os
import uuid



# Define the LLM (Ollama)
llm = ChatOllama(model="qwen2.5:7b", temperature=0.8, num_ctx=8000)


class ChatAgent:
    def __init__(self, llm, system_prompt, tools, storage_dir="chat_logs"):
        self.llm = llm  # Using ChatOllama
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory_store = {}  # Dictionary to store MemorySaver per thread
        self.current_thread_id = "default"
        self.storage_dir = storage_dir  # Directory for saving chat history
        os.makedirs(storage_dir, exist_ok=True)  # Ensure storage directory exists
        self.memory = self.get_memory(self.current_thread_id)
        self.agent_executor = self.create_agent()

    def get_memory(self, thread_id):
        """Retrieves or creates a MemorySaver instance for the given thread_id."""
        if thread_id not in self.memory_store:
            self.memory_store[thread_id] = MemorySaver()
        return self.memory_store[thread_id]

    def create_agent(self):
        """Creates a new agent executor with the current memory."""
        return create_react_agent(
            self.llm, self.tools, checkpointer=self.memory, state_modifier=self.system_prompt
        )

    def change_thread(self, new_thread_id):
        """Switches conversation context to a new thread."""
        self.current_thread_id = new_thread_id
        self.memory = self.get_memory(new_thread_id)
        self.agent_executor = self.create_agent()

    def get_thread_filepath(self):
        """Returns the JSON file path for the current thread."""
        return os.path.join(self.storage_dir, f"{self.current_thread_id}.json")

    def save_to_json(self, message, ai_response):
        """Saves conversation history to a JSON file per thread."""
        chat_data = {"messages": []}

        filepath = self.get_thread_filepath()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    chat_data = json.load(file)
                except json.JSONDecodeError:
                    chat_data = {"messages": []}

        # Append the new conversation
        chat_data["messages"].append({"role": "user", "content": message})
        if ai_response:
            chat_data["messages"].append({"role": "assistant", "content": ai_response})

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(chat_data, file, indent=4)

    def load_chat_history(self):
        """Loads chat history for the current thread from JSON."""
        filepath = self.get_thread_filepath()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    return json.load(file).get("messages", [])
                except json.JSONDecodeError:
                    return []
        return []

    def clear_chat(self):
        """Clears conversation history for the current thread."""
        self.memory.storage.clear()
        self.memory.writes.clear()

        filepath = self.get_thread_filepath()
        if os.path.exists(filepath):
            os.remove(filepath)

    def send_message(self, message):
        """Sends a message, saves it to memory & JSON, and returns the response."""
        try:
            human_message = HumanMessage(content=message)
            response = self.agent_executor.invoke(
                {"messages": [human_message]},
                config={"configurable": {"thread_id": self.current_thread_id}},
            )
            ai_response, tool_calls = self.extract_ai_response(response)

            # Save to JSON file
            self.save_to_json(message, ai_response)

            # Save to MemorySaver
            checkpoint_config = {
                "configurable": {
                    "thread_id": self.current_thread_id,
                    "checkpoint_ns": "chat"
                }
            }
            checkpoint_data = {
                "id": str(uuid.uuid4()),
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": ai_response} if ai_response else None
                ],
                "pending_sends": []
            }

            self.memory.put(
                checkpoint_config,
                checkpoint_data,
                metadata={},
                new_versions={}
            )

            return ai_response, tool_calls
        except Exception as e:
            return f"Error: {str(e)}", []

    def extract_ai_response(self, response):
        """Extracts the AI's response and any tool calls."""
        messages = response.get('messages', [])
        ai_message = None
        tool_calls = []

        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage):
                ai_message = message.content or ai_message
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_calls.extend(message.tool_calls)

        return ai_message, tool_calls



# Define input schema
class SimpleInput(BaseModel):
    task: str = Field(description="A simple task to complete.")

# Define the placeholder tool
class SimpleTaskTool(BaseTool):
    name: str = "simple_task"
    description: str = "A simple tool that takes an input task and returns 'Completed'."
    args_schema: Type[BaseModel] = SimpleInput
    return_direct: bool = False

    def _run(self, task: str) -> str:
        """Returns 'Completed' for any given task."""
        print(f"Task '{task}' completeddddddddddddddddd.")
        return f"Task '{task}' completed."

# Example usage with extensive testing
if __name__ == "__main__":
    system_prompt = "You are a helpful AI assistant that can execute python code and has access to wikipedia."
    simple_tool = SimpleTaskTool()
    tools = []  # Add PythonREPL tool

    agent = ChatAgent(llm, system_prompt, tools)
