import gradio as gr
from components.param_manager import ParamManager
from tkinter import Tk, filedialog

# Instantiate ParamManager
param_manager = ParamManager()

# Function to handle directory selection
def get_folder_path(folder_path: str = "") -> str:
    if not isinstance(folder_path, str):
        raise TypeError("folder_path must be a string")

    try:
        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        selected_folder = filedialog.askdirectory(initialdir=folder_path or ".")
        root.destroy()
        return selected_folder or folder_path
    except Exception as e:
        raise RuntimeError(f"Error initializing folder dialog: {e}") from e

# Function to handle directory selection and update the path
def select_directory():
    selected_directory = get_folder_path()
    if selected_directory:
        param_manager.set_param('directory', selected_directory)
        return f"Directory selected: {selected_directory}"
    else:
        return "No directory selected"

# Define Setup Screen Function for OpenAI API
def setup_openai_api(name, model_name, base_url, api_key, use_embeddings, copy_docs, directory, language, send_config):
    param_manager.set_param('user_name', name)
    param_manager.set_param('agent_type', 'OpenAI API')
    param_manager.set_param('model_name', model_name)
    param_manager.set_param('base_url', base_url)
    param_manager.set_param('api_key', api_key)
    param_manager.set_param('use_embeddings', use_embeddings)
    param_manager.set_param('copy_docs', copy_docs)
    param_manager.set_param('directory', directory.replace("Directory selected: ", ""))
    param_manager.set_param('language', language)
    param_manager.set_param('send_config', send_config)
    with open("setup.flag", "w") as f:
        f.write("setup complete with OpenAI API")
    setup_complete_message = f"Setup complete! Welcome, {name}. You selected OpenAI API. Please refresh the page to proceed to the main screen."
    return setup_complete_message

# Define Setup Screen Function for ReAct agent
def setup_react_agent(name, copy_docs, directory, language, send_config):
    param_manager.set_param('user_name', name)
    param_manager.set_param('agent_type', 'ReAct agent')
    param_manager.set_param('copy_docs', copy_docs)
    param_manager.set_param('directory', directory.replace("Directory selected: ", ""))
    param_manager.set_param('language', language)
    param_manager.set_param('send_config', send_config)
    with open("setup.flag", "w") as f:
        f.write("setup complete with ReAct agent")
    setup_complete_message = f"Setup complete! Welcome, {name}. You selected ReAct agent. Please refresh the page to proceed to the main screen."
    return setup_complete_message

# Define Setup Screen Function for LLMChain
def setup_llmchain(name, copy_docs, directory, language, send_config):
    param_manager.set_param('user_name', name)
    param_manager.set_param('agent_type', 'LLMChain')
    param_manager.set_param('copy_docs', copy_docs)
    param_manager.set_param('directory', directory.replace("Directory selected: ", ""))
    param_manager.set_param('language', language)
    param_manager.set_param('send_config', send_config)
    with open("setup.flag", "w") as f:
        f.write("setup complete with LLMChain")
    setup_complete_message = f"Setup complete! Welcome, {name}. You selected LLMChain. Please refresh the page to proceed to the main screen."
    return setup_complete_message

# Function to show hidden elements and save agent type
def select_agent(agent_type):
    param_manager.set_param('agent_type', agent_type)
    if agent_type == "OpenAI API":
        return (
            gr.update(visible=True, label="Selected OpenAI API. Please enter your name:"),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    elif agent_type == "ReAct agent":
        return (
            gr.update(visible=True, label="Selected ReAct agent. Please enter your name:"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    elif agent_type == "LLMChain":
        return (
            gr.update(visible=True, label="Selected LLMChain. Please enter your name:"),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True)
        )

with gr.Blocks(theme=gr.themes.Soft(), css="""
    footer{display:none !important}
    #chatbot { height: 100%; flex-grow: 1; }
    .flex-container { display: flex; flex-wrap: nowrap; width: 100%; }
    .flex-item { flex: 1 1 0; box-sizing: border-box; padding: 2px; }
    .flex-item img { width: 100%; height: auto; max-width: 100%; }
    """) as setup_interface:
    
    gr.Markdown("## Select an LLM Agent type")
    
    agent_type_state = gr.State()
    
    with gr.Row(elem_classes="flex-container"):
        with gr.Column(elem_classes="flex-item", min_width=100):
            gr.Image("github/react_agent.png", show_fullscreen_button=False, show_label=False, show_download_button=False)
            gr.Markdown("### ReAct agent\nDescription: placeholder etc...")
            react_btn = gr.Button("Select ReAct agent")
        
        with gr.Column(elem_classes="flex-item", min_width=100):
            gr.Image("github/llmchain.png", show_fullscreen_button=False, show_label=False, show_download_button=False)
            gr.Markdown("### LLMChain\nDescription: placeholder")
            llmchain_btn = gr.Button("Select LLMChain")
        
        with gr.Column(elem_classes="flex-item", min_width=100):
            gr.Image("github/openai_api.png", show_fullscreen_button=False, show_label=False, show_download_button=False)
            gr.Markdown("### Use OpenAI API\nDescription: placeholder")
            openai_btn = gr.Button("Select OpenAI API")

    name_input = gr.Textbox(label="Enter your name", visible=False)
    model_name_input = gr.Textbox(label="Model name", visible=False)
    base_url_input = gr.Textbox(label="Base URL or endpoint", visible=False)
    api_key_input = gr.Textbox(label="API key", visible=False)
    use_embeddings_checkbox = gr.Checkbox(label="Do you wanna use OpenAI embeddings (not recommended for security and privacy reasons)", visible=False)
    
    copy_docs_checkbox = gr.Checkbox(label="Copy the documents over to database (recommended)", visible=False)
    directory_button = gr.Button("Select Directory", visible=False)
    language_input = gr.Textbox(label="What is your primary language", visible=False)
    send_config_checkbox = gr.Checkbox(label="Send my system config and settings to the Creator (not API or credential information) for improvement of the app?", visible=False)
    
    setup_output = gr.HTML(label="Output", visible=False)
    submit_btn = gr.Button("Complete Setup", visible=False)
    agent_info = gr.Markdown(visible=False)
    
    react_btn.click(fn=lambda: select_agent("ReAct agent"), inputs=None, outputs=[name_input, model_name_input, base_url_input, api_key_input, use_embeddings_checkbox, copy_docs_checkbox, directory_button, language_input, send_config_checkbox, setup_output, submit_btn, agent_type_state, agent_info])
    llmchain_btn.click(fn=lambda: select_agent("LLMChain"), inputs=None, outputs=[name_input, model_name_input, base_url_input, api_key_input, use_embeddings_checkbox, copy_docs_checkbox, directory_button, language_input, send_config_checkbox, setup_output, submit_btn, agent_type_state, agent_info])
    openai_btn.click(fn=lambda: select_agent("OpenAI API"), inputs=None, outputs=[name_input, model_name_input, base_url_input, api_key_input, use_embeddings_checkbox, copy_docs_checkbox, directory_button, language_input, send_config_checkbox, setup_output, submit_btn, agent_type_state, agent_info])
    
    directory_button.click(fn=select_directory, inputs=None, outputs=setup_output)
    submit_btn.click(fn=setup_openai_api, inputs=[name_input, model_name_input, base_url_input, api_key_input, use_embeddings_checkbox, copy_docs_checkbox, setup_output, language_input, send_config_checkbox], outputs=setup_output)
    submit_btn.click(fn=setup_react_agent, inputs=[name_input, copy_docs_checkbox, setup_output, language_input, send_config_checkbox], outputs=setup_output)
    submit_btn.click(fn=setup_llmchain, inputs=[name_input, copy_docs_checkbox, setup_output, language_input, send_config_checkbox], outputs=setup_output)

#setup_interface.launch()