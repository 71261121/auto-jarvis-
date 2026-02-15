#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 3: AI Engine COMPREHENSIVE Test Suite
=================================================================

This test suite covers:
- Basic functionality tests
- Thread safety tests
- Edge case tests
- Input validation tests
- Integration tests
- Performance tests

All tests must pass for Phase 3 to be considered ZERO ERROR.
"""

import sys
import os
import time
import json
import threading
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from core.ai.openrouter_client import (
    OpenRouterClient, FreeModel, ModelCapability, ChatMessage, AIResponse, ConversationContext,
    MODEL_CAPABILITIES, MODEL_CONTEXT, get_client
)
from core.ai.rate_limiter import (
    RateLimiterManager, AdaptiveRateLimiter, TokenBucket, CircuitBreaker, CircuitState, RateLimitConfig,
    get_rate_limiter_manager
)
from core.ai.model_selector import (
    ModelSelector, TaskType, ModelInfo, ModelStatus, TaskDetector, FREE_MODELS,
    get_model_selector
)
from core.ai.response_parser import (
    ResponseParser, StreamingParser, ParsedResponse, ErrorCode, ErrorDetector,
    get_parser
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
        print(f"  ✗ {name}: {error}")
    
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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASIC FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 1: BASIC FUNCTIONALITY TESTS ===")

# OpenRouter Client Tests
print("\n--- OpenRouter Client ---")

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
print("\n--- Rate Limiter ---")

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
print("\n--- Model Selector ---")

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
print("\n--- Response Parser ---")

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

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 2: EDGE CASE TESTS ===")

# Test MODEL_CAPABILITIES completeness
try:
    # FIX VERIFICATION: All FreeModel members should have capability mapping
    missing_models = []
    for model in FreeModel:
        if model not in MODEL_CAPABILITIES:
            missing_models.append(model.value)
    
    if missing_models:
        results.add_fail("MODEL_CAPABILITIES completeness", f"Missing: {missing_models}")
    else:
        results.add_pass("MODEL_CAPABILITIES completeness (all 9 models mapped)")
except Exception as e:
    results.add_fail("MODEL_CAPABILITIES completeness", str(e))

# Test MODEL_CONTEXT completeness
try:
    missing_context = []
    for model in FreeModel:
        if model not in MODEL_CONTEXT:
            missing_context.append(model.value)
    
    if missing_context:
        results.add_fail("MODEL_CONTEXT completeness", f"Missing: {missing_context}")
    else:
        results.add_pass("MODEL_CONTEXT completeness")
except Exception as e:
    results.add_fail("MODEL_CONTEXT completeness", str(e))

# Test TokenBucket negative tokens validation
try:
    bucket = TokenBucket(capacity=10, refill_rate=2.0)
    try:
        bucket.consume(-1)
        results.add_fail("TokenBucket negative tokens", "Should have raised ValueError")
    except ValueError:
        results.add_pass("TokenBucket negative tokens validation")
except Exception as e:
    results.add_fail("TokenBucket negative tokens", str(e))

# Test TokenBucket zero capacity validation
try:
    try:
        bucket = TokenBucket(capacity=0, refill_rate=1.0)
        results.add_fail("TokenBucket zero capacity", "Should have raised ValueError")
    except ValueError:
        results.add_pass("TokenBucket zero capacity validation")
except Exception as e:
    results.add_fail("TokenBucket zero capacity", str(e))

# Test TokenBucket zero refill_rate validation
try:
    try:
        bucket = TokenBucket(capacity=10, refill_rate=0)
        results.add_fail("TokenBucket zero refill_rate", "Should have raised ValueError")
    except ValueError:
        results.add_pass("TokenBucket zero refill_rate validation")
except Exception as e:
    results.add_fail("TokenBucket zero refill_rate", str(e))

# Test empty message handling
try:
    parser = ResponseParser()
    parsed = parser.parse("")
    assert parsed.success == False
    results.add_pass("Empty string parsing")
except Exception as e:
    results.add_fail("Empty string parsing", str(e))

# Test None handling
try:
    parser = ResponseParser()
    parsed = parser.parse(None)
    assert parsed.success == False
    results.add_pass("None parsing")
except Exception as e:
    results.add_fail("None parsing", str(e))

# Test bytes parsing
try:
    parser = ResponseParser()
    parsed = parser.parse(b'{"choices": [{"message": {"content": "test"}}]}')
    assert parsed.success == True
    results.add_pass("Bytes parsing")
except Exception as e:
    results.add_fail("Bytes parsing", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: THREAD SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 3: THREAD SAFETY TESTS ===")

# Test TokenBucket concurrent access
try:
    bucket = TokenBucket(capacity=100, refill_rate=100.0)
    errors = []
    
    def consume_tokens():
        for _ in range(100):
            try:
                bucket.consume(1)
            except Exception as e:
                errors.append(str(e))
    
    threads = [threading.Thread(target=consume_tokens) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("TokenBucket thread safety", f"Errors: {errors}")
    else:
        results.add_pass("TokenBucket thread safety (1000 operations)")
except Exception as e:
    results.add_fail("TokenBucket thread safety", str(e))

# Test CircuitBreaker concurrent access
try:
    cb = CircuitBreaker(failure_threshold=10)
    errors = []
    
    def record_ops():
        for _ in range(50):
            try:
                cb.record_failure()
                cb.record_success()
            except Exception as e:
                errors.append(str(e))
    
    threads = [threading.Thread(target=record_ops) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("CircuitBreaker thread safety", f"Errors: {errors}")
    else:
        results.add_pass("CircuitBreaker thread safety")
except Exception as e:
    results.add_fail("CircuitBreaker thread safety", str(e))

# Test ModelSelector concurrent access
try:
    selector = ModelSelector()
    errors = []
    results_list = []
    
    def select_models():
        for _ in range(20):
            try:
                result = selector.select_for_task(TaskType.CODING)
                results_list.append(result.model_id)
            except Exception as e:
                errors.append(str(e))
    
    threads = [threading.Thread(target=select_models) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("ModelSelector thread safety", f"Errors: {errors}")
    elif len(results_list) != 100:
        results.add_fail("ModelSelector thread safety", f"Missing results: {len(results_list)}/100")
    else:
        results.add_pass("ModelSelector thread safety (100 selections)")
except Exception as e:
    results.add_fail("ModelSelector thread safety", str(e))

# Test global singleton thread safety
try:
    clients = []
    errors = []
    
    def get_clients():
        try:
            # Reset global for test
            import core.ai.openrouter_client as orc
            orc._client = None
            client = get_client(api_key="test-key")
            clients.append(id(client))
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=get_clients) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    unique_clients = len(set(clients))
    if errors and "API key" not in str(errors[0]):  # API key error is expected
        results.add_fail("Global singleton thread safety", f"Errors: {errors}")
    elif unique_clients != 1:
        results.add_fail("Global singleton thread safety", f"Multiple instances: {unique_clients}")
    else:
        results.add_pass("Global singleton thread safety")
except Exception as e:
    results.add_fail("Global singleton thread safety", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 4: INTEGRATION TESTS ===")

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

try:
    manager = RateLimiterManager()
    manager.register('test_endpoint', RateLimitConfig(requests_per_minute=100))
    result = manager.check('test_endpoint')
    assert result.allowed == True
    results.add_pass("Integration: RateLimiterManager")
except Exception as e:
    results.add_fail("Integration: RateLimiterManager", str(e))

try:
    # Test complete workflow
    selector = get_model_selector()
    result = selector.select("Write a function to sort a list")
    
    parser = get_parser()
    mock_response = {
        "choices": [{
            "message": {
                "content": "def sort_list(lst): return sorted(lst)"
            }
        }],
        "usage": {"total_tokens": 20}
    }
    parsed = parser.parse(mock_response)
    
    if result.model_id and parsed.success:
        results.add_pass("Integration: Full workflow")
    else:
        results.add_fail("Integration: Full workflow", "Incomplete workflow")
except Exception as e:
    results.add_fail("Integration: Full workflow", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== SECTION 5: PERFORMANCE TESTS ===")

# Test TokenBucket performance
try:
    bucket = TokenBucket(capacity=1000, refill_rate=100.0)
    start = time.time()
    for _ in range(10000):
        bucket.consume(1)
    elapsed = time.time() - start
    
    if elapsed < 1.0:  # Should be fast
        results.add_pass(f"TokenBucket performance (10000 ops in {elapsed:.3f}s)")
    else:
        results.add_fail("TokenBucket performance", f"Too slow: {elapsed:.3f}s")
except Exception as e:
    results.add_fail("TokenBucket performance", str(e))

# Test ModelSelector performance
try:
    selector = ModelSelector()
    start = time.time()
    for _ in range(1000):
        selector.select_for_task(TaskType.CODING)
    elapsed = time.time() - start
    
    if elapsed < 2.0:  # Should be reasonably fast
        results.add_pass(f"ModelSelector performance (1000 selections in {elapsed:.3f}s)")
    else:
        results.add_fail("ModelSelector performance", f"Too slow: {elapsed:.3f}s")
except Exception as e:
    results.add_fail("ModelSelector performance", str(e))

# Test ResponseParser performance
try:
    parser = ResponseParser()
    test_response = {"choices": [{"message": {"content": "test" * 100}}]}
    start = time.time()
    for _ in range(10000):
        parser.parse(test_response)
    elapsed = time.time() - start
    
    if elapsed < 2.0:
        results.add_pass(f"ResponseParser performance (10000 parses in {elapsed:.3f}s)")
    else:
        results.add_fail("ResponseParser performance", f"Too slow: {elapsed:.3f}s")
except Exception as e:
    results.add_fail("ResponseParser performance", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
success = results.summary()

if success:
    print("\n" + "🎉 ALL TESTS PASSED! PHASE 3 IS ZERO ERROR!")
else:
    print("\n" + "⚠️ SOME TESTS FAILED - FIXES REQUIRED")

sys.exit(0 if success else 1)
