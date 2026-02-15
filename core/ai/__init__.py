#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - AI Provider Package
=========================================

AI provider modules for JARVIS AI system.

Modules:
    - openrouter_client: OpenRouter API client
    - rate_limiter: Rate limiting and circuit breaker
    - model_selector: Intelligent model selection
    - response_parser: API response parsing

Exports:
    - OpenRouterClient: OpenRouter API client
    - RateLimiterManager: Rate limiting manager
    - ModelSelector: Model selection engine
    - ResponseParser: Response parser
"""

from .openrouter_client import (
    OpenRouterClient,
    FreeModel,
    ModelCapability,
    ChatMessage,
    AIResponse,
    ConversationContext,
    get_client,
    initialize_client,
)

from .rate_limiter import (
    RateLimiterManager,
    AdaptiveRateLimiter,
    TokenBucket,
    CircuitBreaker,
    CircuitState,
    RateLimitConfig,
    RateLimitResult,
    get_rate_limiter_manager,
    rate_limited,
)

from .model_selector import (
    ModelSelector,
    TaskType,
    ModelCapability as SelectionCapability,
    ModelInfo,
    ModelStatus,
    SelectionResult,
    TaskProfile,
    TaskDetector,
    get_model_selector,
    select_model,
)

from .response_parser import (
    ResponseParser,
    StreamingParser,
    ParsedResponse,
    StreamChunk,
    ErrorCode,
    ResponseType,
    ErrorDetector,
    get_parser,
    parse_response,
)

__all__ = [
    # OpenRouter Client
    'OpenRouterClient',
    'FreeModel',
    'ModelCapability',
    'ChatMessage',
    'AIResponse',
    'ConversationContext',
    'get_client',
    'initialize_client',
    
    # Rate Limiter
    'RateLimiterManager',
    'AdaptiveRateLimiter',
    'TokenBucket',
    'CircuitBreaker',
    'CircuitState',
    'RateLimitConfig',
    'RateLimitResult',
    'get_rate_limiter_manager',
    'rate_limited',
    
    # Model Selector
    'ModelSelector',
    'TaskType',
    'ModelInfo',
    'ModelStatus',
    'SelectionResult',
    'TaskProfile',
    'TaskDetector',
    'get_model_selector',
    'select_model',
    
    # Response Parser
    'ResponseParser',
    'StreamingParser',
    'ParsedResponse',
    'StreamChunk',
    'ErrorCode',
    'ResponseType',
    'ErrorDetector',
    'get_parser',
    'parse_response',
]
