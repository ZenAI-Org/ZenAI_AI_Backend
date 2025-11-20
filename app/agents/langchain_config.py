"""
LangChain framework initialization and configuration.
Handles OpenAI API setup, model initialization, and common utilities.
"""

import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LangChainConfig:
    """Configuration for LangChain framework."""
    
    # Default model configuration
    DEFAULT_MODEL = "gpt-4"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TIMEOUT = 300
    
    @staticmethod
    def validate_openai_api_key() -> str:
        """
        Validate and retrieve OpenAI API key from environment.
        
        Returns:
            OpenAI API key
            
        Raises:
            ValueError: If API key is not configured
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please configure your OpenAI API key."
            )
        return api_key
    
    @staticmethod
    def get_model_name() -> str:
        """
        Get configured model name from environment or use default.
        
        Returns:
            Model name to use
        """
        return os.getenv("OPENAI_MODEL_NAME", LangChainConfig.DEFAULT_MODEL)
    
    @staticmethod
    def get_temperature() -> float:
        """
        Get configured temperature from environment or use default.
        
        Returns:
            Temperature value
        """
        try:
            temp = float(os.getenv("OPENAI_TEMPERATURE", LangChainConfig.DEFAULT_TEMPERATURE))
            if not 0.0 <= temp <= 2.0:
                logger.warning(f"Temperature {temp} out of range, using default")
                return LangChainConfig.DEFAULT_TEMPERATURE
            return temp
        except ValueError:
            logger.warning("Invalid temperature value, using default")
            return LangChainConfig.DEFAULT_TEMPERATURE
    
    @staticmethod
    def get_max_tokens() -> Optional[int]:
        """
        Get configured max tokens from environment or use default.
        
        Returns:
            Max tokens value or None
        """
        try:
            max_tokens = os.getenv("OPENAI_MAX_TOKENS")
            if max_tokens:
                return int(max_tokens)
        except ValueError:
            logger.warning("Invalid max_tokens value")
        return LangChainConfig.DEFAULT_MAX_TOKENS
    
    @staticmethod
    def get_timeout() -> int:
        """
        Get configured timeout from environment or use default.
        
        Returns:
            Timeout in seconds
        """
        try:
            timeout = int(os.getenv("OPENAI_TIMEOUT", LangChainConfig.DEFAULT_TIMEOUT))
            if timeout <= 0:
                logger.warning("Timeout must be positive, using default")
                return LangChainConfig.DEFAULT_TIMEOUT
            return timeout
        except ValueError:
            logger.warning("Invalid timeout value, using default")
            return LangChainConfig.DEFAULT_TIMEOUT


class LangChainInitializer:
    """Initializes and manages LangChain components."""
    
    _llm_instance: Optional[ChatOpenAI] = None
    _initialized: bool = False
    
    @classmethod
    def initialize(cls) -> ChatOpenAI:
        """
        Initialize LangChain with OpenAI API.
        
        Returns:
            Initialized ChatOpenAI instance
            
        Raises:
            ValueError: If OpenAI API key is not configured
        """
        if cls._initialized and cls._llm_instance:
            logger.info("LangChain already initialized, returning existing instance")
            return cls._llm_instance
        
        try:
            # Validate API key
            api_key = LangChainConfig.validate_openai_api_key()
            
            # Get configuration
            model_name = LangChainConfig.get_model_name()
            temperature = LangChainConfig.get_temperature()
            max_tokens = LangChainConfig.get_max_tokens()
            timeout = LangChainConfig.get_timeout()
            
            logger.info(f"Initializing LangChain with model: {model_name}")
            
            # Initialize ChatOpenAI
            cls._llm_instance = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                api_key=api_key,
            )
            
            cls._initialized = True
            logger.info("LangChain successfully initialized")
            
            return cls._llm_instance
        
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {str(e)}")
            raise
    
    @classmethod
    def get_llm(cls) -> ChatOpenAI:
        """
        Get the initialized LLM instance.
        
        Returns:
            ChatOpenAI instance
            
        Raises:
            RuntimeError: If LangChain has not been initialized
        """
        if not cls._initialized or not cls._llm_instance:
            raise RuntimeError(
                "LangChain has not been initialized. "
                "Call LangChainInitializer.initialize() first."
            )
        return cls._llm_instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the LangChain instance (useful for testing)."""
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
        """
        Create a LangChain prompt template.
        
        Args:
            template: Template string with {variable} placeholders
            input_variables: List of variable names
            description: Optional template description
            
        Returns:
            PromptTemplate instance
        """
        return PromptTemplate(
            template=template,
            input_variables=input_variables,
            description=description,
        )
    
    @staticmethod
    def format_prompt(template: PromptTemplate, **kwargs) -> str:
        """
        Format a prompt template with values.
        
        Args:
            template: PromptTemplate instance
            **kwargs: Values for template variables
            
        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)


class MessageBuilder:
    """Utility for building LangChain message objects."""
    
    @staticmethod
    def create_system_message(content: str) -> SystemMessage:
        """
        Create a system message.
        
        Args:
            content: Message content
            
        Returns:
            SystemMessage instance
        """
        return SystemMessage(content=content)
    
    @staticmethod
    def create_human_message(content: str) -> HumanMessage:
        """
        Create a human message.
        
        Args:
            content: Message content
            
        Returns:
            HumanMessage instance
        """
        return HumanMessage(content=content)
    
    @staticmethod
    def create_messages(
        system_prompt: Optional[str] = None,
        user_message: str = "",
    ) -> list:
        """
        Create a list of messages for LLM invocation.
        
        Args:
            system_prompt: Optional system prompt
            user_message: User message content
            
        Returns:
            List of message objects
        """
        messages = []
        if system_prompt:
            messages.append(MessageBuilder.create_system_message(system_prompt))
        messages.append(MessageBuilder.create_human_message(user_message))
        return messages
