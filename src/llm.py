from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from src.config import settings

def get_llm(temperature=0.2):
    """Returns the appropriate chat model with automatic high-throughput fallbacks."""
    if settings.is_cloud_mode:
        # 1. Primary Model: 3.5 Flash
        primary_llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_LLM_MODEL,  # "gemini-3.5-flash"
            google_api_key=settings.GEMINI_API_KEY,
            temperature=temperature,
            max_retries=0
        )
        
        # 2. Secondary: 3.1 Flash-Lite
        fb_1 = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite", 
            google_api_key=settings.GEMINI_API_KEY, 
            temperature=temperature,
            max_retries=0
        )
        
        # 3. Tertiary: 2.5 Flash-Lite
        fb_2 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            google_api_key=settings.GEMINI_API_KEY, 
            temperature=temperature,
            max_retries=0
        )
        
        # 4. Final Catch: 2.5 Flash
        fb_3 = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=settings.GEMINI_API_KEY, 
            temperature=temperature,
            max_retries=0
        )
        
        # Chain the models together to effectively quadruple the rate limit
        return primary_llm.with_fallbacks([fb_1, fb_2, fb_3])
    else:
        return ChatOllama(model=settings.OLLAMA_MODEL_NAME, temperature=temperature)
