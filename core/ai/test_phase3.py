#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 3: AI Engine Comprehensive Test Suite
"""

import sys
import os
import time
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from core.ai.openrouter_client import (
    OpenRouterClient, FreeModel, ModelCapability, ChatMessage, AIResponse, ConversationContext
)
from core.ai.rate_limiter import (
    RateLimiterManager, AdaptiveRateLimiter, TokenBucket, CircuitBreaker, CircuitState, RateLimitConfig
)
from core.ai.model_selector import (
    ModelSelector, TaskType, ModelInfo, ModelStatus, TaskDetector, FREE_MODELS
)
from core.ai.response_parser import (
    ResponseParser, StreamingParser, ParsedResponse, ErrorCode, ErrorDetector
)

class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.start_time = time.time()
    
    def add_pass(self, name):
        self.passed.append(name)
        print(f"  ✓ {name}")
    
    def add_fail(self, name, error):
        self.failed.append((name, error))
        print(f"  ✗ {name}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed)
        rate = len(self.passed) / total * 100 if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"PHASE 3 AI ENGINE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {len(self.passed)} | Failed: {len(self.failed)}")
        print(f"Success Rate: {rate:.1f}% | Time: {elapsed:.2f}s")
        print(f"{'='*60}")
        return len(self.failed) == 0

results = TestResult()

# OpenRouter Client Tests
print("\n=== OPENROUTER CLIENT TESTS ===")

try:
    assert FreeModel.AUTO_FREE.value == "openrouter/free"
    results.add_pass("FreeModel enum")
except Exception as e:
    results.add_fail("FreeModel enum", str(e))

try:
    msg = ChatMessage(role="user", content="Hello")
    assert msg.to_dict()["role"] == "user"
    results.add_pass("ChatMessage creation")
except Exception as e:
    results.add_fail("ChatMessage creation", str(e))

try:
    response = AIResponse(content="Hi", model="test", success=True)
    assert response.success == True
    results.add_pass("AIResponse creation")
except Exception as e:
    results.add_fail("AIResponse creation", str(e))

try:
    ctx = ConversationContext(conversation_id="test")
    ctx.add_message("user", "Hello")
    assert len(ctx.to_messages()) == 1
    results.add_pass("ConversationContext")
except Exception as e:
    results.add_fail("ConversationContext", str(e))

# Rate Limiter Tests
print("\n=== RATE LIMITER TESTS ===")

try:
    bucket = TokenBucket(capacity=10, refill_rate=2.0)
    assert bucket.get_tokens() == 10
    results.add_pass("TokenBucket init")
except Exception as e:
    results.add_fail("TokenBucket init", str(e))

try:
    bucket = TokenBucket(capacity=10, refill_rate=2.0)
    result = bucket.consume(1)
    assert result.allowed == True
    results.add_pass("TokenBucket consume")
except Exception as e:
    results.add_fail("TokenBucket consume", str(e))

try:
    bucket = TokenBucket(capacity=5, refill_rate=1.0)
    for _ in range(5):
        bucket.consume(1)
    result = bucket.consume(1)
    assert result.allowed == False
    results.add_pass("TokenBucket exhaustion")
except Exception as e:
    results.add_fail("TokenBucket exhaustion", str(e))

try:
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED
    results.add_pass("CircuitBreaker init")
except Exception as e:
    results.add_fail("CircuitBreaker init", str(e))

try:
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    results.add_pass("CircuitBreaker open")
except Exception as e:
    results.add_fail("CircuitBreaker open", str(e))

try:
    config = RateLimitConfig(requests_per_minute=60)
    limiter = AdaptiveRateLimiter(config)
    result = limiter.check()
    assert result.allowed == True
    results.add_pass("AdaptiveRateLimiter check")
except Exception as e:
    results.add_fail("AdaptiveRateLimiter check", str(e))

# Model Selector Tests
print("\n=== MODEL SELECTOR TESTS ===")

try:
    assert hasattr(TaskType, 'CODING')
    assert hasattr(TaskType, 'REASONING')
    results.add_pass("TaskType enum")
except Exception as e:
    results.add_fail("TaskType enum", str(e))

try:
    detector = TaskDetector()
    profile = detector.detect("Write a Python function")
    assert profile.task_type == TaskType.CODING
    results.add_pass("Coding detection")
except Exception as e:
    results.add_fail("Coding detection", str(e))

try:
    detector = TaskDetector()
    profile = detector.detect("Explain why the sky is blue")
    assert profile.task_type == TaskType.REASONING
    results.add_pass("Reasoning detection")
except Exception as e:
    results.add_fail("Reasoning detection", str(e))

try:
    selector = ModelSelector()
    result = selector.select_for_task(TaskType.CODING)
    assert result.model_id is not None
    results.add_pass("Model selection for coding")
except Exception as e:
    results.add_fail("Model selection for coding", str(e))

try:
    selector = ModelSelector()
    available = selector.get_available_models()
    assert len(available) > 0
    results.add_pass("Get available models")
except Exception as e:
    results.add_fail("Get available models", str(e))

try:
    assert "openrouter/free" in FREE_MODELS
    results.add_pass("FREE_MODELS dict")
except Exception as e:
    results.add_fail("FREE_MODELS dict", str(e))

# Response Parser Tests
print("\n=== RESPONSE PARSER TESTS ===")

try:
    parser = ResponseParser()
    response = {"choices": [{"message": {"content": "Hello"}}]}
    parsed = parser.parse(response)
    assert parsed.success == True
    results.add_pass("Parse success response")
except Exception as e:
    results.add_fail("Parse success response", str(e))

try:
    parser = ResponseParser()
    response = {"error": {"message": "Rate limit exceeded"}}
    parsed = parser.parse(response)
    assert parsed.success == False
    results.add_pass("Parse error response")
except Exception as e:
    results.add_fail("Parse error response", str(e))

try:
    parser = ResponseParser()
    parsed = parser.parse("invalid json")
    assert parsed.success == False
    results.add_pass("Parse invalid JSON")
except Exception as e:
    results.add_fail("Parse invalid JSON", str(e))

try:
    code, msg = ErrorDetector.detect_error({"error": {"message": "Invalid API key"}})
    assert code == ErrorCode.INVALID_API_KEY
    results.add_pass("ErrorDetector")
except Exception as e:
    results.add_fail("ErrorDetector", str(e))

try:
    sp = StreamingParser()
    chunk = sp.feed_line('data: {"choices":[{"delta":{"content":"Hi"}}]}')
    assert chunk.content == "Hi"
    results.add_pass("StreamingParser")
except Exception as e:
    results.add_fail("StreamingParser", str(e))

# Integration Tests
print("\n=== INTEGRATION TESTS ===")

try:
    selector = ModelSelector()
    parser = ResponseParser()
    result = selector.select_for_task(TaskType.CODING)
    mock_response = {"choices": [{"message": {"content": "code"}}]}
    parsed = parser.parse(mock_response)
    assert parsed.success == True
    results.add_pass("Integration: Selector-Parser")
except Exception as e:
    results.add_fail("Integration: Selector-Parser", str(e))

# Summary
success = results.summary()
print("\n" + ("🎉 ALL TESTS PASSED!" if success else "⚠️ SOME TESTS FAILED"))
sys.exit(0 if success else 1)
