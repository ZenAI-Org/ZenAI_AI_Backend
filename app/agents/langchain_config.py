"""
LangChain framework initialization and configuration.
Handles OpenAI, Groq, and Google Gemini API setup, model initialization, and common utilities.
"""

import os
import logging
from typing import Optional, Union, Any
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LangChainConfig:
    """Configuration for LangChain framework."""
    
    # Default model configuration
    DEFAULT_PROVIDER = "openai"  # "openai", "groq", or "gemini"
    DEFAULT_MODEL_OPENAI = "gpt-4"
    DEFAULT_MODEL_GROQ = "llama-3.3-70b-versatile"
    DEFAULT_MODEL_GEMINI = "gemini-1.5-pro-latest"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TIMEOUT = 300
    
    @staticmethod
    def get_provider() -> str:
        """
        Get configured LLM provider.
        Priority:
        1. LLM_PROVIDER env var
        2. "groq" if GROQ_API_KEY is set and no others
        3. "gemini" if GOOGLE_API_KEY is set and no others
        4. "openai" (default)
        """
        provider = os.getenv("LLM_PROVIDER")
        if provider:
            return provider.lower()
            
        # Auto-detection logic (simple)
        if os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            return "groq"
        if os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            return "gemini"
            
        return LangChainConfig.DEFAULT_PROVIDER

    @staticmethod
    def validate_api_key(provider: str) -> str:
        """
        Validate and retrieve API key for the specified provider.
        """
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            return api_key
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            return api_key
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set.")
            return api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def get_model_name(provider: str) -> str:
        """Get configured model name based on provider."""
        if provider == "groq":
            return os.getenv("GROQ_MODEL_NAME", LangChainConfig.DEFAULT_MODEL_GROQ)
        elif provider == "gemini":
            return os.getenv("GEMINI_MODEL_NAME", LangChainConfig.DEFAULT_MODEL_GEMINI)
        return os.getenv("OPENAI_MODEL_NAME", LangChainConfig.DEFAULT_MODEL_OPENAI)
    
    @staticmethod
    def get_temperature() -> float:
        """Get configured temperature from environment or use default."""
        try:
            temp = float(os.getenv("LLM_TEMPERATURE", str(LangChainConfig.DEFAULT_TEMPERATURE)))
            if not 0.0 <= temp <= 2.0:
                logger.warning(f"Temperature {temp} out of range, using default")
                return LangChainConfig.DEFAULT_TEMPERATURE
            return temp
        except ValueError:
            logger.warning("Invalid temperature value, using default")
            return LangChainConfig.DEFAULT_TEMPERATURE
    
    @staticmethod
    def get_max_tokens() -> Optional[int]:
        """Get configured max tokens from environment or use default."""
        try:
            max_tokens = os.getenv("LLM_MAX_TOKENS")
            if max_tokens:
                return int(max_tokens)
        except ValueError:
            logger.warning("Invalid max_tokens value")
        return LangChainConfig.DEFAULT_MAX_TOKENS
    
    @staticmethod
    def get_timeout() -> int:
        """Get configured timeout from environment or use default."""
        try:
            timeout = int(os.getenv("LLM_TIMEOUT", str(LangChainConfig.DEFAULT_TIMEOUT)))
            if timeout <= 0:
                logger.warning("Timeout must be positive, using default")
                return LangChainConfig.DEFAULT_TIMEOUT
            return timeout
        except ValueError:
            logger.warning("Invalid timeout value, using default")
            return LangChainConfig.DEFAULT_TIMEOUT


class LangChainInitializer:
    """Initializes and manages LangChain components."""
    
    _llm_instance: Optional[Union[ChatOpenAI, ChatGroq, Any]] = None
    _initialized: bool = False
    
    @classmethod
    def initialize(cls) -> Union[ChatOpenAI, ChatGroq, Any]:
        """
        Initialize LangChain with configured provider.
        """
        if cls._initialized and cls._llm_instance:
            logger.info("LangChain already initialized, returning existing instance")
            return cls._llm_instance
        
        try:
            # Determine provider
            provider = LangChainConfig.get_provider()
            logger.info(f"Initializing LangChain with provider: {provider}")
            
            # Validate API key
            api_key = LangChainConfig.validate_api_key(provider)
            
            # Get configuration
            model_name = LangChainConfig.get_model_name(provider)
            temperature = LangChainConfig.get_temperature()
            max_tokens = LangChainConfig.get_max_tokens()
            timeout = LangChainConfig.get_timeout()
            
            if provider == "groq":
                cls._llm_instance = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            elif provider == "gemini":
                if ChatGoogleGenerativeAI is None:
                    raise ImportError("langchain-google-genai package not installed. Run 'pip install langchain-google-genai'")
                    
                cls._llm_instance = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout=timeout,
                    convert_system_message_to_human=True # Gemini sometimes prefers this
                )
            else:
                cls._llm_instance = ChatOpenAI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            
            cls._initialized = True
            logger.info(f"LangChain successfully initialized with {model_name}")
            
            return cls._llm_instance
        
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {str(e)}")
            raise
    
    @classmethod
    def get_llm(cls) -> Union[ChatOpenAI, ChatGroq, Any]:
        """Get the initialized LLM instance."""
        if not cls._initialized or not cls._llm_instance:
            # Auto-initialize if not ready
            return cls.initialize()
        return cls._llm_instance

    
    @classmethod
    def reset(cls) -> None:
        """Reset the LangChain instance."""
        cls._llm_instance = None
        cls._initialized = False
        logger.info("LangChain instance reset")


class PromptTemplateManager:
    """Manages reusable prompt templates for agents."""
    
    @staticmethod
    def create_template(
        template: str,
        input_variables: list,
        description: Optional[str] = None,
    ) -> PromptTemplate:
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            description=description,
        )
    
    @staticmethod
    def format_prompt(template: PromptTemplate, **kwargs) -> str:
        return template.format(**kwargs)


class MessageBuilder:
    """Utility for building LangChain message objects."""
    
    @staticmethod
    def create_system_message(content: str) -> SystemMessage:
        return SystemMessage(content=content)
    
    @staticmethod
    def create_human_message(content: str) -> HumanMessage:
        return HumanMessage(content=content)
    
    @staticmethod
    def create_messages(
        system_prompt: Optional[str] = None,
        user_message: str = "",
    ) -> list:
        messages = []
        if system_prompt:
            messages.append(MessageBuilder.create_system_message(system_prompt))
        messages.append(MessageBuilder.create_human_message(user_message))
        return messages
