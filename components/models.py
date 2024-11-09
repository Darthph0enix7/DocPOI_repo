from components.param_manager import ParamManager
import os

# Initialize ParamManager
param_manager = ParamManager()
params = param_manager.get_all_params()

# Extract required parameters
agent_type = params.get('agent_type')

def setup_models():
    llm = None
    embeddings = None

    # Conditional logic based on agent_type
    if agent_type == "OpenAI API":
        embed_model = params.get('embed_model', "text-embedding-3-small")
        base_url = params.get("base_url")
        api_key = params.get('api_key')
        model_name = params.get('model_name', "gpt4o-mini")
        
        from langchain_openai import OpenAIEmbeddings
        from langchain_openai import ChatOpenAI

        embeddings = OpenAIEmbeddings(
            model=embed_model,
            base_url=base_url,
            api_key=api_key,
        )

        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            max_retries=5,
        )

    elif agent_type == "LLMChain":
        embed_model = params.get('embed_model')
        local_model = params.get("local_model", "Llama3.1 8b")

        # Translate local_model based on its value
        model_translation = {
            "Llama3.1 8b": "llama3.1-8b",
            "Qwen 2.5 7b": "qwen-2.5-7b",
            "gemma2 9b": "gemma2-9b"
        }

        translated_local_model = model_translation.get(local_model, local_model)

        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=translated_local_model,
        )

        if embed_model is None:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model=translated_local_model,
            )
        
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model=embed_model,
            )

    elif agent_type == "ReAct agent":
        embed_model = params.get('embed_model')
        local_model = params.get("local_model", "Mistral Nemo 12B")

        # Translate local_model based on its value
        model_translation = {
            "Mistral Nemo 12B": "mistral-nemo",
            "Qwen 2.5 14b": "qwen-2.5-14b",
            "gemma2 9b": "gemma2-9b"
        }

        translated_local_model = model_translation.get(local_model, local_model)

        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=translated_local_model,
        )

        if embed_model is None:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(
                model=translated_local_model,
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model=embed_model,
            )
    else:
        print("Unknown agent type")

    return llm, embeddings