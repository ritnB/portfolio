# llm/providers.py - LLM Provider abstraction
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
import json
from loguru import logger

from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    OPENAI_API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, FREE_MODEL_FALLBACK
)


class BaseLLMProvider(ABC):
    """Base LLM Provider class"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check availability"""
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
            
            # Temporary debugging: Use completely hardcoded prompt
            import re
            
            # Analyze original prompt
            logger.info(f"Original prompt length: {len(prompt)}")
            logger.info(f"Original prompt preview: {repr(prompt[:50])}")
            
            # Find problematic characters
            problem_chars = []
            for i, char in enumerate(prompt):
                if ord(char) > 127:
                    problem_chars.append(f"pos {i}: '{char}' (ord: {ord(char)})")
            
            if problem_chars:
                logger.warning(f"Non-ASCII characters found: {problem_chars[:10]}")  # Max 10
            
            # Use simple English prompt for testing
            prompt = "Write a short cryptocurrency market analysis post. Keep it under 100 characters and include relevant emojis."
            
            logger.info(f"Using test prompt: {prompt}")
            logger.info(f"Test prompt length: {len(prompt)}")
            
            # ASCII validation
            try:
                test_ascii = prompt.encode('ascii')
                logger.info(f"ASCII encoding successful: {len(test_ascii)} bytes")
            except UnicodeEncodeError as e:
                logger.error(f"Test prompt ASCII encoding failed: {e}")
                prompt = "Write a crypto post under 100 chars."
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
            
            # API key cleanup and safety check
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
            api_key_clean = re.sub(r'[^\w\-]', '', raw_api_key.strip())  # Allow only alphanumeric and hyphens
            
            logger.info(f"Cleaned API key length: {len(api_key_clean)}")
            logger.info(f"Cleaned API key first 10 chars: {api_key_clean[:10]}...")
            
            # Check if cleaned API key is in correct format
            if not api_key_clean or api_key_clean == "NONE":
                logger.warning("API key not configured - running in test mode")
                # Return test dummy response
                return "AI analysis shows cryptocurrency market volatility. Data-driven trend analysis completed. #crypto #AI"
            
            if not api_key_clean.startswith('sk-'):
                logger.error(f"API key format error: {api_key_clean[:20]}...")
                raise Exception("OpenAI API key is not in correct format. Must start with 'sk-'.")
            
            if len(api_key_clean) < 40:
                logger.error(f"API key length insufficient: {len(api_key_clean)} chars")
                raise Exception("OpenAI API key is too short.")
            
            # Generate ASCII-safe header
            try:
                auth_header = f"Bearer {api_key_clean}"
                auth_header.encode('ascii')  # ASCII test
                logger.info("Authorization header ASCII encoding successful")
            except UnicodeEncodeError as e:
                logger.error(f"Cleaned API key ASCII encoding failed: {e}")
                # Last resort: complete ASCII filtering
                ascii_only_key = ''.join(c for c in api_key_clean if ord(c) < 128)
                auth_header = f"Bearer {ascii_only_key}"
                logger.warning(f"Forced ASCII filtering applied: {len(ascii_only_key)} chars")
            
            headers = {
                "Authorization": auth_header,
                "Content-Type": "application/json"
            }
            
            logger.info(f"Request headers generated")
            
            # Use requests session for safer processing
            try:
                session = requests.Session()
                session.headers.update(headers)
                
                logger.info("Starting request transmission...")
                
                response = session.post(
                    "https://api.openai.com/v1/chat/completions",
                    data=json_bytes,
                    timeout=30
                )
                logger.info(f"Request completed: {response.status_code}")
                
            except Exception as e:
                logger.error(f"Session request failed: {e}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error details: {repr(e)}")
                
                # Alternative: direct bytes processing
                try:
                    logger.info("Trying alternative method: complete manual processing")
                    
                    import urllib.request
                    import urllib.error
                    
                    req = urllib.request.Request(
                        "https://api.openai.com/v1/chat/completions",
                        data=json_bytes,
                        headers={
                            b"Authorization": auth_header.encode('ascii'),
                            b"Content-Type": b"application/json"
                        }
                    )
                    
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        response_data = json.loads(resp.read().decode('utf-8'))
                        logger.info("urllib method successful!")
                        
                        # Process response
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            result = response_data["choices"][0]["message"]["content"]
                            if result:
                                result = result.strip()
                            else:
                                result = ""
                        else:
                            result = ""
                        
                        logger.info(f"Response received successfully: {len(result)} characters")
                        return result
                        
                except Exception as e2:
                    logger.error(f"urllib alternative also failed: {e2}")
                    raise Exception(f"All HTTP methods failed: {e}, {e2}")
            
            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            # Safe response processing
            if "choices" in response_data and len(response_data["choices"]) > 0:
                result = response_data["choices"][0]["message"]["content"]
                if result:
                    result = result.strip()
                else:
                    result = ""
            else:
                result = ""
            
            logger.debug(f"Response received successfully: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def is_available(self) -> bool:
        # OpenAI requires API key
        return bool(self.api_key and len(self.api_key.strip()) > 20)


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face model provider"""
    
    def __init__(self):
        self.api_key = HUGGINGFACE_API_KEY
        self.model = HUGGINGFACE_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            # Optimize prompt (for HuggingFace free API)
            if isinstance(prompt, str):
                # Enhance prompt for target language response
                optimized_prompt = f"{prompt}\n\nPlease respond in target language with appropriate emojis:"
                optimized_prompt = optimized_prompt.encode('utf-8').decode('utf-8')
                logger.info(f"HuggingFace prompt length: {len(optimized_prompt)}")
            else:
                optimized_prompt = prompt
            
            # Use API key if available, otherwise free access
            headers = {
                "Content-Type": "application/json; charset=utf-8"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                logger.info("Using HuggingFace API key")
            else:
                logger.info("Using HuggingFace free API")
            
            payload = {
                "inputs": optimized_prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": min(self.max_tokens, 150),  # Free API has limitations
                    "return_full_text": False,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"HuggingFace API error: {response.text}")
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "").strip()
            else:
                text = str(result).strip()
            
            # Check result encoding
            if isinstance(text, str):
                text = text.encode('utf-8').decode('utf-8')
            
            return text
            
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def is_available(self) -> bool:
        # HuggingFace also provides free API, so always available
        return True


class OllamaProvider(BaseLLMProvider):
    """Ollama local model provider"""
    
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.temperature = LLM_TEMPERATURE
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            # Check prompt encoding
            if isinstance(prompt, str):
                prompt = prompt.encode('utf-8').decode('utf-8')
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            
            result = response.json()
            text = result.get("response", "").strip()
            
            # Check result encoding
            if isinstance(text, str):
                text = text.encode('utf-8').decode('utf-8')
            
            return text
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class FallbackProvider(BaseLLMProvider):
    """Free model fallback provider"""
    
    def __init__(self):
        # Free model priority (lazy loading)
        self.provider_classes = [
            ("ollama", OllamaProvider),
            ("huggingface", HuggingFaceProvider),
            ("openai", OpenAIProvider)  # Last resort
        ]
    
    def generate(self, prompt: str, **kwargs) -> str:
        last_error = None
        
        logger.info("FallbackProvider started - trying free models...")
        
        for provider_name, provider_class in self.provider_classes:
            logger.info(f"📋 Checking {provider_name} availability...")
            
            try:
                # Create instance only when needed (lazy loading)
                provider = provider_class()
                
                if provider.is_available():
                    logger.info(f"✅ {provider_name} available - attempting...")
                    try:
                        logger.info(f"🤖 Using {provider_name} model...")
                        result = provider.generate(prompt, **kwargs)
                        logger.success(f"✅ {provider_name} successful!")
                        return result
                    except Exception as e:
                        logger.warning(f"❌ {provider_name} failed: {e}")
                        last_error = e
                        continue
                else:
                    logger.info(f"❌ {provider_name} unavailable - skipping")
            except Exception as e:
                logger.warning(f"❌ {provider_name} instance creation failed: {e}")
                last_error = e
                continue
        
        logger.error("All providers failed")
        if last_error:
            raise last_error
        else:
            raise Exception("No available LLM providers.")
    
    def is_available(self) -> bool:
        # True if at least one provider is available
        for provider_name, provider_class in self.provider_classes:
            try:
                provider = provider_class()
                if provider.is_available():
                    return True
            except Exception:
                continue
        return False


def get_llm_provider() -> BaseLLMProvider:
    """Return LLM provider based on configuration"""
    
    logger.info(f"🔧 Checking LLM provider configuration:")
    logger.info(f"   FREE_MODEL_FALLBACK: {FREE_MODEL_FALLBACK}")
    logger.info(f"   LLM_PROVIDER: {LLM_PROVIDER}")
    
    if FREE_MODEL_FALLBACK:
        logger.info("✅ FREE_MODEL_FALLBACK=True -> Using FallbackProvider")
        return FallbackProvider()
    
    logger.info("❌ FREE_MODEL_FALLBACK=False -> Using individual provider")
    
    if LLM_PROVIDER == "openai":
        logger.info(f"Selected OpenAIProvider")
        return OpenAIProvider()
    elif LLM_PROVIDER == "huggingface":
        logger.info(f"Selected HuggingFaceProvider")
        return HuggingFaceProvider()
    elif LLM_PROVIDER == "ollama":
        logger.info(f"Selected OllamaProvider")
        return OllamaProvider()
    else:
        logger.warning(f"Unknown provider: {LLM_PROVIDER}, using fallback")
        return FallbackProvider()


def generate_text(prompt: str, **kwargs) -> str:
    """Unified text generation function"""
    provider = get_llm_provider()
    
    if not provider.is_available():
        raise Exception("No available LLM models.")
    
    # Safe prompt processing
    try:
        if isinstance(prompt, str):
            prompt = prompt.encode('utf-8', errors='replace').decode('utf-8')
    except Exception as e:
        logger.warning(f"Error during prompt encoding processing: {e}")
        # Safe fallback
        import re
        prompt = re.sub(r'[^\w\s가-힣.,!?%\-\n]', '', prompt)
    
    return provider.generate(prompt, **kwargs) 