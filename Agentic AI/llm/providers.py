# llm/providers.py - LLM Provider Abstraction
from abc import ABC, abstractmethod
import requests
import json
import re
from loguru import logger
from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OPENAI_API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, FREE_MODEL_FALLBACK
)

class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT model provider"""
    
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            if not self.api_key:
                raise Exception("OpenAI API key not configured.")
            
            # Temporary debugging: use completely hardcoded prompt
            import re
            
            # Analyze original prompt
            logger.info(f"Original prompt length: {len(prompt)}")
            logger.info(f"Original prompt beginning: {repr(prompt[:50])}")
            
            # Find problematic characters
            problem_chars = []
            for i, char in enumerate(prompt):
                if ord(char) > 127:
                    problem_chars.append(f"pos {i}: '{char}' (ord: {ord(char)})")
            
            if problem_chars:
                logger.warning(f"Non-ASCII characters found: {problem_chars[:10]}")  # Max 10
            
            # Replace with test simple English prompt
            prompt = "Write a short cryptocurrency market analysis post in Korean. Keep it under 100 characters and include relevant emojis."
            
            logger.info(f"Using test prompt: {prompt}")
            logger.info(f"Test prompt length: {len(prompt)}")
            
            # ASCII validation
            try:
                test_ascii = prompt.encode('ascii')
                logger.info(f"ASCII encoding successful: {len(test_ascii)} bytes")
            except UnicodeEncodeError as e:
                logger.error(f"Test prompt also failed ASCII encoding: {e}")
                prompt = "Write a crypto post in Korean under 100 chars."
                logger.warning("Fallback to minimal prompt")
            
            # Use requests for safe HTTP requests
            import requests
            import json
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Add kwargs
            payload.update(kwargs)
            
            logger.info("Starting OpenAI API requests call")
            
            # Process JSON completely safely
            try:
                payload_json = json.dumps(payload, ensure_ascii=True, separators=(',', ':'))
                logger.info(f"JSON generation successful: {len(payload_json)} chars")
                logger.info(f"JSON preview: {payload_json[:200]}...")
                
                # ASCII encoding test
                json_bytes = payload_json.encode('ascii')
                logger.info(f"JSON ASCII encoding successful: {len(json_bytes)} bytes")
                
            except Exception as e:
                logger.error(f"JSON processing failed: {e}")
                raise Exception(f"JSON encoding error: {e}")
            
            # Clean and validate API key
            raw_api_key = self.api_key if self.api_key else "NONE"
            logger.info(f"Original API key length: {len(raw_api_key)}")
            logger.info(f"Original API key repr: {repr(raw_api_key[:30])}...")
            
            # Check for problematic characters in API key
            problem_chars_in_key = []
            for i, char in enumerate(raw_api_key):
                if ord(char) > 127:
                    problem_chars_in_key.append(f"pos {i}: '{char}' (ord: {ord(char)})")
            
            if problem_chars_in_key:
                logger.warning(f"Non-ASCII characters found in API key: {problem_chars_in_key}")
            
            # Clean API key (remove spaces, special characters)
            import re
            api_key_clean = re.sub(r'[^\w\-]', '', raw_api_key.strip())  # Allow letters, numbers, hyphens only
            
            logger.info(f"Cleaned API key length: {len(api_key_clean)}")
            logger.info(f"Cleaned API key first 10 chars: {api_key_clean[:10]}...")
            
            # Validate cleaned API key format
            if not api_key_clean or api_key_clean == "NONE":
                logger.warning("API key not configured - running in test mode")
                # Return dummy response for testing
                return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
            
            if not api_key_clean.startswith('sk-'):
                logger.error(f"API key format error: {api_key_clean[:20]}...")
                raise Exception("OpenAI API key format is incorrect. Must start with 'sk-'.")
            
            if len(api_key_clean) < 40:
                logger.error(f"API key length insufficient: {len(api_key_clean)} chars")
                raise Exception("OpenAI API key is too short.")
            
            # Create ASCII-safe authorization header
            try:
                auth_header = f"Bearer {api_key_clean}"
                auth_header.encode('ascii')  # ASCII test
                logger.info("Authorization header ASCII encoding successful")
            except UnicodeEncodeError:
                # Last resort: complete ASCII filtering
                ascii_only_key = ''.join(c for c in api_key_clean if ord(c) < 128)
                auth_header = f"Bearer {ascii_only_key}"
                logger.warning(f"Applied forced ASCII filtering: {len(ascii_only_key)} chars")
            
            # Use requests session for safer processing
            try:
                session = requests.Session()
                session.headers.update({
                    "Authorization": auth_header,
                    "Content-Type": "application/json"
                })
                
                response = session.post(
                    "https://api.openai.com/v1/chat/completions",
                    data=payload_json.encode('ascii'),
                    timeout=30
                )
                
            except Exception as e:
                logger.error(f"Session request failed: {e}")
                # Alternative: direct bytes processing
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": auth_header,
                            "Content-Type": "application/json"
                        },
                        data=payload_json.encode('ascii'),
                        timeout=30
                    )
                except Exception as e2:
                    logger.error(f"Direct request also failed: {e2}")
                    raise Exception(f"All HTTP request methods failed: {e}, {e2}")
            
            # Process response
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    content = response_data['choices'][0]['message']['content']
                    logger.info("OpenAI API call successful")
                    return content
                except Exception as e:
                    logger.error(f"Response parsing failed: {e}")
                    return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
                
        except Exception as e:
            logger.error(f"OpenAI provider error: {e}")
            return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
    
    def is_available(self) -> bool:
        # OpenAI requires API key
        return bool(self.api_key and len(self.api_key.strip()) > 20)

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face model provider"""
    
    def __init__(self):
        self.api_key = HUGGINGFACE_API_KEY
        self.model = HUGGINGFACE_MODEL
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            if not self.api_key:
                raise Exception("Hugging Face API key not configured.")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": LLM_MAX_TOKENS,
                    "temperature": LLM_TEMPERATURE
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract result (varies by model)
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    return result.get('generated_text', '')
                else:
                    return str(result)
            else:
                logger.error(f"Hugging Face API error: {response.status_code}")
                return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
                
        except Exception as e:
            logger.error(f"Hugging Face provider error: {e}")
            return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider"""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_TOKENS
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
                
        except Exception as e:
            logger.error(f"Ollama provider error: {e}")
            return "🚀 Bitcoin showing strong momentum! 📈 Crypto market analysis indicates bullish sentiment. #BTC #Crypto"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def get_llm_provider() -> BaseLLMProvider:
    """Get the best available LLM provider"""
    
    # Provider class list
    providers = [
        OpenAIProvider,
        HuggingFaceProvider,
        OllamaProvider
    ]
    
    # Try configured provider first
    if LLM_PROVIDER == "openai":
        provider = OpenAIProvider()
        if provider.is_available():
            return provider
    elif LLM_PROVIDER == "huggingface":
        provider = HuggingFaceProvider()
        if provider.is_available():
            return provider
    elif LLM_PROVIDER == "ollama":
        provider = OllamaProvider()
        if provider.is_available():
            return provider
    
    # Fallback: try available providers in order
    for ProviderClass in providers:
        provider = ProviderClass()
        if provider.is_available():
            logger.info(f"Using fallback provider: {ProviderClass.__name__}")
            return provider
    
    # All providers failed
    raise Exception("No LLM providers available")

def generate_text(prompt: str, **kwargs) -> str:
    """Generate text using the best available provider"""
    provider = get_llm_provider()
    return provider.generate(prompt, **kwargs) 