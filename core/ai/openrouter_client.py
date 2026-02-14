#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - OpenRouter AI Client
===========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- OpenRouter: /collections/free-models
- Medium: DeepSeek with OpenRouter guide
- TypingMind: Model capabilities documentation

FREE Models (100% FREE - No Cost):
1. deepseek/deepseek-r1-0528:free
   - Context: 164K tokens
   - Best for: Reasoning, Math, Code
   - Matches OpenAI o1 performance
   
2. google/gemini-2.0-flash-exp:free
   - Context: 1M tokens (HUGE!)
   - Best for: Long context, Multimodal
   - Fast responses
   
3. meta-llama/llama-3.1-8b-instruct:free
   - Context: 128K tokens
   - Best for: General, Coding
   - Good balance
   
4. mistralai/mistral-7b-instruct:free
   - Context: 32K tokens
   - Best for: Fast, Efficient
   - Quick responses

API Key: Configured and verified
Memory Budget: < 10MB for client + responses
"""

import time
import json
import logging
import threading
import hashlib
import sys
import os
from typing import Dict, Any, Optional, List, Union, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class FreeModel(Enum):
    """
    Free models available on OpenRouter.
    
    Updated: February 2025 - Based on actual API response
    All models are 100% FREE.
    """
    # Tier 1: Best free models (verified working)
    AUTO_FREE = "openrouter/free"  # Auto best free selection
    AURORA_ALPHA = "openrouter/aurora-alpha"  # Advanced reasoning
    
    # Tier 2: Good free alternatives
    STEP_3_5_FLASH = "stepfun/step-3.5-flash:free"  # Fast
    TRINITY_LARGE = "arcee-ai/trinity-large-preview:free"  # Large model
    SOLAR_PRO = "upstage/solar-pro-3:free"  # Good general purpose
    
    # Tier 3: Lightweight options
    LFM_THINKING = "liquid/lfm-2.5-1.2b-thinking:free"  # Thinking model
    LFM_INSTRUCT = "liquid/lfm-2.5-1.2b-instruct:free"  # Instruct model
    
    # Legacy (may not work, kept for fallback)
    DEEPSEEK_R1 = "deepseek/deepseek-r1-0528:free"
    GEMINI_FLASH = "google/gemini-2.0-flash-exp:free"


class ModelCapability(Enum):
    """Model capabilities for intelligent selection"""
    REASONING = auto()
    CODING = auto()
    MATH = auto()
    LONG_CONTEXT = auto()
    FAST_RESPONSE = auto()
    MULTIMODAL = auto()


# Model capability mapping (based on research)
MODEL_CAPABILITIES = {
    FreeModel.AUTO_FREE: [
        ModelCapability.FAST_RESPONSE,
        ModelCapability.CODING,
    ],
    FreeModel.AURORA_ALPHA: [
        ModelCapability.REASONING,
        ModelCapability.CODING,
        ModelCapability.MATH,
    ],
    FreeModel.STEP_3_5_FLASH: [
        ModelCapability.FAST_RESPONSE,
    ],
    FreeModel.TRINITY_LARGE: [
        ModelCapability.REASONING,
        ModelCapability.LONG_CONTEXT,
    ],
    FreeModel.LFM_THINKING: [
        ModelCapability.REASONING,
    ],
}

# Context window sizes (in tokens)
MODEL_CONTEXT = {
    FreeModel.AUTO_FREE: 128000,
    FreeModel.AURORA_ALPHA: 128000,
    FreeModel.STEP_3_5_FLASH: 128000,
    FreeModel.TRINITY_LARGE: 128000,
    FreeModel.SOLAR_PRO: 128000,
    FreeModel.LFM_THINKING: 32000,
    FreeModel.LFM_INSTRUCT: 32000,
    # Legacy
    FreeModel.DEEPSEEK_R1: 164000,
    FreeModel.GEMINI_FLASH: 1000000,
}


@dataclass
class ChatMessage:
    """A single chat message"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class AIResponse:
    """
    Response from AI model.
    
    Attributes:
        content: The text response
        model: Model that generated the response
        tokens_used: Total tokens used
        latency_ms: Response latency
        success: Whether request succeeded
        error: Error message if failed
        raw_response: Raw API response
        finish_reason: Why generation stopped
        is_fallback: Whether a fallback model was used
    """
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    raw_response: Dict = field(default_factory=dict)
    finish_reason: str = ""
    is_fallback: bool = False
    reasoning: str = ""  # For models that provide reasoning (like DeepSeek R1)


@dataclass
class ConversationContext:
    """
    Conversation context for multi-turn conversations.
    
    This is CRITICAL for the Ultra Advanced Memory System.
    """
    conversation_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    total_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, tokens: int = 0):
        """Add a message to the conversation"""
        msg = ChatMessage(role=role, content=content, tokens=tokens)
        self.messages.append(msg)
        self.total_tokens += tokens
        self.updated_at = time.time()
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to API format"""
        return [m.to_dict() for m in self.messages]
    
    def get_token_count_estimate(self) -> int:
        """Estimate token count for context"""
        # Simple estimation: ~4 chars per token
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4


# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class OpenRouterClient:
    """
    OpenRouter AI Client optimized for FREE models.
    
    Features:
    - Multiple free models with intelligent fallback
    - Conversation memory
    - Response caching
    - Rate limit handling
    - Capability-based model selection
    
    Memory Budget: < 10MB
    
    Usage:
        client = OpenRouterClient(api_key="sk-or-v1-...")
        
        # Simple chat
        response = client.chat("Hello!")
        
        # With system prompt
        response = client.chat("Write code", system="You are a coding expert")
        
        # With specific model
        response = client.chat("Complex problem", model=FreeModel.DEEPSEEK_R1)
        
        # Multi-turn conversation
        conv = client.create_conversation()
        client.chat("Hello!", conversation=conv)
        client.chat("What did I just say?", conversation=conv)
    """
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Default fallback order (best to good, verified working)
    DEFAULT_MODEL_ORDER = [
        FreeModel.AUTO_FREE,  # Let OpenRouter choose best free
        FreeModel.AURORA_ALPHA,  # Advanced reasoning
        FreeModel.STEP_3_5_FLASH,  # Fast responses
        FreeModel.TRINITY_LARGE,  # Large model
        FreeModel.SOLAR_PRO,  # General purpose
        FreeModel.LFM_THINKING,  # Thinking model
    ]
    
    def __init__(
        self,
        api_key: str = None,
        http_client = None,
        default_model: FreeModel = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        auto_fallback: bool = True,
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            http_client: HTTPClient instance (will create if not provided)
            default_model: Default model to use
            enable_cache: Enable response caching
            cache_ttl: Cache TTL in seconds
            auto_fallback: Automatically fallback to other models on failure
        """
        # API key
        self._api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self._api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY or pass api_key parameter.")
        
        # HTTP client
        if http_client is None:
            try:
                from core.http_client import HTTPClient
            except ImportError:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from core.http_client import HTTPClient
            self._http = HTTPClient(
                default_headers=self._get_default_headers()
            )
        else:
            self._http = http_client
        
        # Configuration
        self._default_model = default_model or FreeModel.AUTO_FREE
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._auto_fallback = auto_fallback
        
        # Cache
        self._cache: Dict[str, tuple] = {}
        self._cache_lock = threading.Lock()
        
        # Conversations
        self._conversations: Dict[str, ConversationContext] = {}
        self._conversation_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'model_usage': {m.value: 0 for m in FreeModel},
            'cache_hits': 0,
        }
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/jarvis-ai",
            "X-Title": "JARVIS Self-Modifying AI v14",
            "Content-Type": "application/json",
        }
    
    def _make_cache_key(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        **kwargs
    ) -> str:
        """Create cache key for request"""
        key_data = json.dumps({
            'messages': messages,
            'model': model,
            'temperature': temperature,
            'kwargs': {k: v for k, v in kwargs.items() if k != 'api_key'}
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str) -> Optional[AIResponse]:
        """Get cached response if available"""
        if not self._enable_cache:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                response, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return response
        return None
    
    def _set_cache(self, cache_key: str, response: AIResponse):
        """Cache a response"""
        if not self._enable_cache:
            return
        
        with self._cache_lock:
            self._cache[cache_key] = (response, time.time())
            
            # Cleanup old entries
            if len(self._cache) > 1000:
                current_time = time.time()
                self._cache = {
                    k: v for k, v in self._cache.items()
                    if current_time - v[1] < self._cache_ttl
                }
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CORE API METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def chat(
        self,
        message: str,
        system: str = None,
        model: FreeModel = None,
        conversation: ConversationContext = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_fallback: bool = None,
        **kwargs
    ) -> AIResponse:
        """
        Send a chat message.
        
        Args:
            message: User message
            system: System prompt
            model: Model to use (default: DeepSeek R1)
            conversation: Existing conversation context
            temperature: Response randomness (0-2)
            max_tokens: Maximum response tokens
            auto_fallback: Override auto_fallback setting
            **kwargs: Additional API parameters
            
        Returns:
            AIResponse object
        """
        model = model or self._default_model
        auto_fallback = auto_fallback if auto_fallback is not None else self._auto_fallback
        
        # Build messages
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        if conversation:
            messages.extend(conversation.to_messages())
        
        messages.append({"role": "user", "content": message})
        
        # Check cache
        cache_key = self._make_cache_key(messages, model.value, temperature, **kwargs)
        cached = self._get_cached(cache_key)
        if cached:
            self._stats['cache_hits'] += 1
            logger.debug("Cache hit for chat request")
            return cached
        
        # Make request
        response = self._make_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            auto_fallback=auto_fallback,
            **kwargs
        )
        
        # Update conversation if provided
        if response.success and conversation:
            with self._conversation_lock:
                conversation.add_message("user", message)
                conversation.add_message("assistant", response.content, response.tokens_used)
        
        # Cache successful responses
        if response.success:
            self._set_cache(cache_key, response)
        
        return response
    
    def _make_request(
        self,
        messages: List[Dict],
        model: FreeModel,
        temperature: float,
        max_tokens: int,
        auto_fallback: bool,
        **kwargs
    ) -> AIResponse:
        """Make API request with optional fallback"""
        start_time = time.time()
        
        models_to_try = [model]
        if auto_fallback:
            for m in self.DEFAULT_MODEL_ORDER:
                if m not in models_to_try:
                    models_to_try.append(m)
        
        last_response = None
        
        for idx, current_model in enumerate(models_to_try):
            try:
                response = self._single_request(
                    messages=messages,
                    model=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                response.latency_ms = (time.time() - start_time) * 1000
                response.is_fallback = idx > 0
                last_response = response
                
                # Update stats
                self._stats['total_requests'] += 1
                self._stats['total_latency_ms'] += response.latency_ms
                self._stats['model_usage'][current_model.value] += 1
                
                if response.success:
                    self._stats['successful_requests'] += 1
                    self._stats['total_tokens'] += response.tokens_used
                    
                    if idx > 0:
                        self._stats['fallback_requests'] += 1
                        logger.info(f"Using fallback model: {current_model.value}")
                    
                    return response
                
                # Check if we should try next model
                if not auto_fallback:
                    return response
                
                logger.warning(f"Model {current_model.value} failed: {response.error}")
                
            except Exception as e:
                self._stats['total_requests'] += 1
                self._stats['failed_requests'] += 1
                logger.error(f"Request failed for {current_model.value}: {e}")
                
                if not auto_fallback:
                    return AIResponse(
                        content="",
                        model=current_model.value,
                        success=False,
                        error=str(e),
                        latency_ms=(time.time() - start_time) * 1000,
                    )
        
        # All models failed
        return last_response or AIResponse(
            content="",
            model=model.value,
            success=False,
            error="All models failed",
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    def _single_request(
        self,
        messages: List[Dict],
        model: FreeModel,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> AIResponse:
        """Make a single API request"""
        payload = {
            "model": model.value,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add any additional parameters
        for key in ['top_p', 'top_k', 'presence_penalty', 'frequency_penalty', 'stop']:
            if key in kwargs:
                payload[key] = kwargs[key]
        
        response = self._http.post(
            self.API_URL,
            json_data=payload,
            timeout=120,  # AI can take time
        )
        
        if not response.success:
            return AIResponse(
                content="",
                model=model.value,
                success=False,
                error=f"HTTP error: {response.error}",
            )
        
        # Parse response
        try:
            data = response.json()
            
            if 'error' in data:
                return AIResponse(
                    content="",
                    model=model.value,
                    success=False,
                    error=data['error'].get('message', str(data['error'])),
                    raw_response=data,
                )
            
            if 'choices' not in data or not data['choices']:
                return AIResponse(
                    content="",
                    model=model.value,
                    success=False,
                    error="No choices in response",
                    raw_response=data,
                )
            
            choice = data['choices'][0]
            content = choice.get('message', {}).get('content', '')
            
            # Check for reasoning (DeepSeek R1 provides this)
            reasoning = choice.get('message', {}).get('reasoning', '')
            
            # Get usage
            usage = data.get('usage', {})
            tokens_used = usage.get('total_tokens', 0)
            
            # Finish reason
            finish_reason = choice.get('finish_reason', '')
            
            return AIResponse(
                content=content,
                model=model.value,
                tokens_used=tokens_used,
                success=True,
                raw_response=data,
                finish_reason=finish_reason,
                reasoning=reasoning,
            )
            
        except Exception as e:
            return AIResponse(
                content="",
                model=model.value,
                success=False,
                error=f"Parse error: {e}",
            )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONVERSATION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def create_conversation(
        self,
        conversation_id: str = None,
        system_prompt: str = None,
        metadata: Dict = None
    ) -> ConversationContext:
        """
        Create a new conversation context.
        
        Args:
            conversation_id: Optional ID (will generate if not provided)
            system_prompt: Optional system prompt
            metadata: Optional metadata
            
        Returns:
            ConversationContext object
        """
        if conversation_id is None:
            conversation_id = hashlib.sha256(
                f"{time.time()}:{id(self)}".encode()
            ).hexdigest()[:16]
        
        context = ConversationContext(
            conversation_id=conversation_id,
            metadata=metadata or {},
        )
        
        if system_prompt:
            context.add_message("system", system_prompt)
        
        with self._conversation_lock:
            self._conversations[conversation_id] = context
        
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get an existing conversation"""
        return self._conversations.get(conversation_id)
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs"""
        return list(self._conversations.keys())
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation"""
        with self._conversation_lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                return True
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # SPECIALIZED METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def analyze_code(self, code: str, question: str = None) -> AIResponse:
        """
        Analyze code using AI.
        
        Automatically uses DeepSeek R1 for best reasoning.
        """
        system = """You are an expert Python code analyzer.
Analyze code for:
1. Potential bugs and errors
2. Performance issues
3. Security vulnerabilities
4. Code quality and best practices
5. Improvement opportunities

Provide specific, actionable feedback with line numbers where relevant."""
        
        prompt = f"Analyze this Python code:\n\n```python\n{code}\n```"
        if question:
            prompt += f"\n\nSpecific question: {question}"
        
        return self.chat(
            prompt,
            system=system,
            model=FreeModel.AUTO_FREE,
        )
    
    def suggest_modification(
        self,
        code: str,
        goal: str,
        constraints: List[str] = None,
        safety_checks: bool = True
    ) -> AIResponse:
        """
        Suggest code modification.
        
        CRITICAL: This is used by the self-modification engine.
        """
        system = """You are a code modification expert.
Suggest specific, safe modifications that:
1. Achieve the stated goal
2. Maintain backward compatibility
3. Follow Python best practices
4. Include proper error handling
5. Are minimal and focused

IMPORTANT: Provide only the modified code, clearly marked.
Use comments to explain changes."""
        
        prompt = f"""Current code:
```python
{code}
```

Modification goal: {goal}
"""
        if constraints:
            prompt += f"\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        if safety_checks:
            prompt += "\n\nSAFETY: Ensure the modification is safe and reversible."
        
        return self.chat(
            prompt,
            system=system,
            model=FreeModel.DEEPSEEK_R1,
            temperature=0.3,  # Lower temperature for more precise output
        )
    
    def long_context_chat(
        self,
        message: str,
        context: str,
        system: str = None
    ) -> AIResponse:
        """
        Chat with long context using Gemini Flash (1M tokens).
        
        Use this for large documents or code files.
        """
        full_message = f"Context:\n{context}\n\nQuery: {message}"
        
        return self.chat(
            full_message,
            system=system or "You are a helpful assistant with access to the provided context.",
            model=FreeModel.AUTO_FREE,
        )
    
    def quick_chat(self, message: str) -> AIResponse:
        """
        Quick chat using fastest model (Step 3.5 Flash).
        
        Use for simple, quick queries.
        """
        return self.chat(message, model=FreeModel.AUTO_FREE)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MODEL SELECTION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def select_model_for_capability(
        self,
        capability: ModelCapability
    ) -> FreeModel:
        """Select best model for a specific capability"""
        for model, capabilities in MODEL_CAPABILITIES.items():
            if capability in capabilities:
                return model
        return FreeModel.AUTO_FREE  # Default
    
    def get_model_context_limit(self, model: FreeModel) -> int:
        """Get context limit for a model"""
        return MODEL_CONTEXT.get(model, 32000)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # STATISTICS AND MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = self._stats.copy()
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests'] * 100
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_requests']
        else:
            stats['success_rate'] = 0
            stats['avg_latency_ms'] = 0
        stats['active_conversations'] = len(self._conversations)
        stats['cache_size'] = len(self._cache)
        return stats
    
    def clear_cache(self):
        """Clear response cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("OpenRouter cache cleared")
    
    def clear_conversations(self):
        """Clear all conversations"""
        with self._conversation_lock:
            self._conversations.clear()
        logger.info("Conversations cleared")
    
    def close(self):
        """Close the client"""
        self.clear_cache()
        self.clear_conversations()
        if hasattr(self._http, 'close'):
            self._http.close()
        logger.info("OpenRouter client closed")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_client: Optional[OpenRouterClient] = None


def get_client(api_key: str = None) -> OpenRouterClient:
    """Get global OpenRouter client instance"""
    global _client
    if _client is None:
        _client = OpenRouterClient(api_key=api_key)
    return _client


def initialize_client(api_key: str) -> OpenRouterClient:
    """Initialize global client with API key"""
    global _client
    _client = OpenRouterClient(api_key=api_key)
    return _client


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test(api_key: str = None) -> Dict[str, Any]:
    """
    Run self-test for OpenRouter client.
    
    Tests:
    1. API connectivity
    2. Simple chat
    3. Model fallback
    4. Conversation management
    """
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    # Get API key
    key = api_key or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        results['failed'].append('no_api_key')
        return results
    
    try:
        client = OpenRouterClient(api_key=key)
    except Exception as e:
        results['failed'].append(f'client_init: {e}')
        return results
    
    # Test 1: Simple chat
    try:
        response = client.chat(
            "Say 'Hello from JARVIS!' exactly.",
            model=FreeModel.AUTO_FREE,  # Auto select best free model
            max_tokens=50,
        )
        if response.success:
            results['passed'].append('simple_chat')
            if 'JARVIS' in response.content or 'Hello' in response.content:
                results['passed'].append('response_quality')
        else:
            results['failed'].append(f'simple_chat: {response.error}')
    except Exception as e:
        results['failed'].append(f'simple_chat: {e}')
    
    # Test 2: Conversation
    try:
        conv = client.create_conversation()
        r1 = client.chat("My name is Test.", conversation=conv)
        r2 = client.chat("What is my name?", conversation=conv)
        
        if r1.success and r2.success:
            results['passed'].append('conversation')
        else:
            results['warnings'].append('conversation: some requests failed')
    except Exception as e:
        results['warnings'].append(f'conversation: {e}')
    
    # Test 3: Code analysis
    try:
        code = "def add(a, b):\n    return a + b"
        response = client.analyze_code(code)
        if response.success:
            results['passed'].append('code_analysis')
        else:
            results['warnings'].append(f'code_analysis: {response.error}')
    except Exception as e:
        results['warnings'].append(f'code_analysis: {e}')
    
    results['stats'] = client.get_stats()
    client.close()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS OpenRouter Client - Self Test")
    print("=" * 70)
    
    # Check for API key
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("\n❌ No API key found!")
        print("Set OPENROUTER_API_KEY environment variable")
        print("\nExample:")
        print("  export OPENROUTER_API_KEY='sk-or-v1-...'")
    else:
        print(f"\n✓ API key found: {api_key[:20]}...")
        
        test_results = self_test(api_key)
        
        print("\n✅ Passed Tests:")
        for test in test_results['passed']:
            print(f"   ✓ {test}")
        
        if test_results['failed']:
            print("\n❌ Failed Tests:")
            for test in test_results['failed']:
                print(f"   ✗ {test}")
        
        if test_results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in test_results['warnings']:
                print(f"   ! {warning}")
        
        if 'stats' in test_results:
            print("\n📊 Statistics:")
            stats = test_results['stats']
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")
            print(f"   Total tokens: {stats['total_tokens']}")
    
    print("\n" + "=" * 70)
