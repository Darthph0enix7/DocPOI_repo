import gradio as gr
from components.param_manager import ParamManager

# Initialize ParamManager
param_manager = ParamManager()
params = param_manager.get_all_params()

# Function to update parameter
def update_param(param_name, value):
    param_manager.set_param(param_name, value)

def create_settings_interface():
    with gr.Tab("Settings"):
        # Agent type dropdown
        agent_type = gr.Dropdown(
            choices=["OpenAI API", "LLMChain", "ReAct agent"],
            value=params.get("agent_type", "OpenAI API"),
            label="Agent Type",
            interactive=True
        )
        agent_type.change(lambda val: update_param("agent_type", val), agent_type, None)

        # Username input
        username = gr.Textbox(
            value=params.get("user_name", ""),
            label="Username",
            interactive=True
        )
        username.change(lambda val: update_param("user_name", val), username, None)

        # Copy Docs Checkbox
        copy_docs = gr.Checkbox(
            value=params.get("copy_docs", False),
            label="Copy Docs",
            interactive=True
        )
        copy_docs.change(lambda val: update_param("copy_docs", val), copy_docs, None)

        # Directory Input
        directory = gr.Textbox(
            value=params.get("directory", ""),
            label="Directory",
            interactive=True
        )
        directory.change(lambda val: update_param("directory", val), directory, None)

        # Language Input
        language = gr.Textbox(
            value=params.get("language", ""),
            label="Language",
            interactive=True
        )
        language.change(lambda val: update_param("language", val), language, None)

        # Send Config Checkbox
        send_config = gr.Checkbox(
            value=params.get("send_config", False),
            label="Send Config",
            interactive=True
        )
        send_config.change(lambda val: update_param("send_config", val), send_config, None)

        # Conditional settings based on agent type
        model_name = gr.Textbox(
            value=params.get("model_name", ""),
            label="Model Name",
            visible=params.get("agent_type") == "OpenAI API",
            interactive=True
        )
        model_name.change(lambda val: update_param("model_name", val), model_name, None)

        base_url = gr.Textbox(
            value=params.get("base_url", ""),
            label="Base URL",
            visible=params.get("agent_type") == "OpenAI API",
            interactive=True
        )
        base_url.change(lambda val: update_param("base_url", val), base_url, None)

        api_key = gr.Textbox(
            value=params.get("api_key", ""),
            label="API Key",
            visible=params.get("agent_type") == "OpenAI API",
            interactive=True
        )
        api_key.change(lambda val: update_param("api_key", val), api_key, None)

        use_embeddings = gr.Checkbox(
            value=params.get("use_embeddings", False),
            label="Use Embeddings",
            visible=params.get("agent_type") == "OpenAI API",
            interactive=True
        )
        use_embeddings.change(lambda val: update_param("use_embeddings", val), use_embeddings, None)

        embed_model_openai = gr.Dropdown(
            choices=["text-embedding-3-small", "text-embedding-3-large"],
            value=params.get("embed_model", params.get("embed_model", "text-embedding-3-small")),
            label="Embed Model",
            visible=params.get("agent_type") == "OpenAI API" and params.get("use_embeddings", False),
            interactive=True
        )
        embed_model_openai.change(lambda val: update_param("embed_model", val), embed_model_openai, None)

        local_model_llmchain = gr.Dropdown(
            choices=["Llama3.1 8b", "Qwen 2.5 7b", "gemma2 9b"],
            value=params.get("local_model", "Llama3.1 8b"),
            label="Local Model",
            visible=params.get("agent_type") == "LLMChain",
            interactive=True
        )
        local_model_llmchain.change(lambda val: update_param("local_model", val), local_model_llmchain, None)

        embed_model_llmchain = gr.Textbox(
            value=params.get("embed_model", ""),
            label="Embed Model",
            visible=params.get("agent_type") == "LLMChain",
            interactive=True
        )
        embed_model_llmchain.change(lambda val: update_param("embed_model", val), embed_model_llmchain, None)

        local_model_react = gr.Dropdown(
            choices=["Mistral Nemo 12B", "Qwen 2.5 14b", "gemma2 9b"],
            value=params.get("local_model", "Mistral Nemo 12B"),
            label="Local Model",
            visible=params.get("agent_type") == "ReAct agent",
            interactive=True
        )
        local_model_react.change(lambda val: update_param("local_model", val), local_model_react, None)

        embed_model_react = gr.Textbox(
            value=params.get("embed_model", ""),
            label="Embed Model",
            visible=params.get("agent_type") == "ReAct agent",
            interactive=True
        )
        embed_model_react.change(lambda val: update_param("embed_model", val), embed_model_react, None)

        def update_visibility(agent_type, use_embeddings):
            return (
                gr.update(visible=agent_type == "OpenAI API"),
                gr.update(visible=agent_type == "OpenAI API"),
                gr.update(visible=agent_type == "OpenAI API"),
                gr.update(visible=agent_type == "OpenAI API"),
                gr.update(visible=agent_type == "OpenAI API" and use_embeddings),
                gr.update(visible=agent_type == "LLMChain"),
                gr.update(visible=agent_type == "LLMChain"),
                gr.update(visible=agent_type == "ReAct agent"),
                gr.update(visible=agent_type == "ReAct agent"),
            )

        agent_type.change(
            update_visibility,
            inputs=[agent_type, use_embeddings],
            outputs=[
                model_name,
                base_url,
                api_key,
                use_embeddings,
                embed_model_openai,
                local_model_llmchain,
                embed_model_llmchain,
                local_model_react,
                embed_model_react
            ]
        )

        use_embeddings.change(
            lambda val: gr.update(visible=val),
            inputs=[use_embeddings],
            outputs=[embed_model_openai]
        )


