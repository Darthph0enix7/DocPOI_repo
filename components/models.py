from components.param_manager import ParamManager
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize ParamManager
param_manager = ParamManager()
params = param_manager.get_all_params()

# Extract required parameters
agent_type = params.get('agent_type')

def setup_models(temperature=0.9, num_predict=8000):
    llm = None
    embeddings = None

    # Conditional logic based on agent_type
    if agent_type == "OpenAI API":
        embed_model = params.get('embed_model', "text-embedding-3-small")
        base_url = params.get("base_url")
        api_key = params.get('api_key')
        model_name = params.get('model_name', "gpt4o-mini")
        max_tokens = num_predict 
        
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
            max_tokens=max_tokens,
            temperature=temperature
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

        llm = ChatOllama(
            model=translated_local_model,
            temperature=temperature,
            num_predict=num_predict
        )

        if embed_model is None:

            embeddings = OllamaEmbeddings(
                model=translated_local_model,
            )
        
        else:

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

        llm = ChatOllama(
            model=translated_local_model,
            temperature=temperature,
            num_predict=num_predict
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