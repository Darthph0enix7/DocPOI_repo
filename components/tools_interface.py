import gradio as gr
import subprocess
import ast
import os

# Hardcoded examples of tools
prewritten_examples = {
    "Firebase Tool": """
import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    cred = credentials.Certificate("path/to/your/firebase/credentials.json")
    firebase_admin.initialize_app(cred)

    return "Firebase has been initialized!"
""",
    "Wikipedia Query Tool": """
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Example function using the tool
def search_wikipedia(query):
    return wikipedia_tool.run(query)
"""
}

general_tool = prewritten_examples["Firebase Tool"]

# Define a function to validate Python syntax
def validate_syntax(code):
    try:
        ast.parse(code)
        return "Syntax is valid!"
    except SyntaxError as e:
        return f"Syntax error: {e}"

# Function to install pip packages
def install_packages(packages):
    if packages:
        for package in packages.split(','):
            package = package.strip()
            if package:
                subprocess.run(["pip", "install", package])

# Function to save the tool code to a file and update saved tools list
def save_tool(tool_name, code, packages):
    if not tool_name:
        return "Error: Tool name cannot be empty."
    
    # Validate the syntax before saving
    syntax_check = validate_syntax(code)
    if "Syntax error" in syntax_check:
        return syntax_check
    
    # Create a directory for the tools if it doesn't exist
    if not os.path.exists("user_tools"):
        os.makedirs("user_tools")
    
    # Save the code to a Python file
    tool_path = os.path.join("user_tools", f"{tool_name}.py")
    with open(tool_path, "w") as f:
        f.write(code)
    
    # Install the required packages
    install_packages(packages)
    
    # Update saved tools list after saving
    update_saved_tools()
    
    return f"Tool '{tool_name}' has been saved successfully!"

# Function to load saved tools
def load_saved_tools():
    tools = {}
    if os.path.exists("user_tools"):
        for filename in os.listdir("user_tools"):
            if filename.endswith(".py"):
                tool_name = filename[:-3]
                with open(os.path.join("user_tools", filename), "r") as f:
                    tools[tool_name] = f.read()
    return tools

# Function to update saved tools dropdown
saved_tools = load_saved_tools()

def update_saved_tools():
    global saved_tools
    saved_tools = load_saved_tools()

# Gradio interface
def create_tools_interface():
    with gr.Row():
        with gr.Column(scale=20):
            gr.Markdown("# Custom Tool Creator for Chatbot Agent")
        with gr.Column(scale=1, min_width=1):
            info_btn = gr.Button("",icon="github/189664.png", link="https://eren.enpoi.com/#:~:text=Balancing%20professional%20excellence%20with%20personal%20growth%2C%20I%20am%20committed%20to%20continuous")
    
    # Code editor
    code_editor = gr.Code(label="Tool Code", language="python", lines=20)
    
    # Load a general example by default
    code_editor.value = general_tool
    
    with gr.Row():
        with gr.Column():
            # Tool name input
            tool_name = gr.Textbox(label="Tool Name", placeholder="Enter the name of your tool")       
            # Pip packages input
            pip_packages = gr.Textbox(label="Pip Packages", placeholder="e.g., firebase-admin, requests")
            
        with gr.Column():
            # List of saved tools
            saved_tools_dropdown = gr.Dropdown(choices=list(saved_tools.keys()), label="Your Saved Tools")
            
            # Function to load saved tool code
            def load_tool_code(tool_name):
                if isinstance(tool_name, list):
                    tool_name = tool_name[0]  # Get the first element if it's a list
                tool_code = saved_tools.get(tool_name, "")
                return tool_code, tool_name
            
            saved_tools_dropdown.change(load_tool_code, inputs=saved_tools_dropdown, outputs=[code_editor, tool_name])
            
            # Dropdown for selecting examples
            def load_example(example_name):
                return prewritten_examples.get(example_name, "")
            
            example_dropdown = gr.Dropdown(choices=list(prewritten_examples.keys()), label="Load Example Tool")
            example_dropdown.change(load_example, inputs=example_dropdown, outputs=code_editor)
    
    # Save tool button
    save_btn = gr.Button("Save Tool")
    save_result = gr.Textbox(label="Save Result", interactive=False)
    save_btn.click(save_tool, inputs=[tool_name, code_editor, pip_packages], outputs=save_result)
    
    # Update saved tools dropdown after saving
    save_btn.click(lambda: list(saved_tools.keys()), None, saved_tools_dropdown)
