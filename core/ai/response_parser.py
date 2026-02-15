#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - API Response Parser
=========================================

Device: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Research-Based Implementation:
- OpenRouter API response format
- Streaming response handling
- SSE (Server-Sent Events) parsing
- Chunked transfer encoding
- Error response normalization

Features:
- Streaming response support
- Chunk parsing for large responses
- Error detection and extraction
- Token usage tracking
- Reasoning extraction (DeepSeek R1)
- Memory-efficient chunk processing

Memory Impact: < 1MB for streaming buffer
"""

import json
import time
import logging
import threading
import re
from typing import Dict, Any, Optional, List, Generator, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from io import StringIO

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseType(Enum):
    """Type of API response"""
    SUCCESS = auto()
    ERROR = auto()
    STREAMING = auto()
    PARTIAL = auto()
    RATE_LIMITED = auto()
    TIMEOUT = auto()


class ErrorCode(Enum):
    """Common API error codes"""
    UNKNOWN = auto()
    INVALID_API_KEY = auto()
    RATE_LIMIT = auto()
    CONTEXT_TOO_LONG = auto()
    MODEL_NOT_FOUND = auto()
    CONTENT_FILTERED = auto()
    SERVER_ERROR = auto()
    TIMEOUT = auto()
    NETWORK_ERROR = auto()
    QUOTA_EXCEEDED = auto()


@dataclass
class ParsedResponse:
    """
    Parsed API response with all metadata.
    
    Unified response format regardless of API differences.
    """
    # Core content
    content: str = ""
    reasoning: str = ""  # For reasoning models like DeepSeek R1
    
    # Status
    success: bool = True
    response_type: ResponseType = ResponseType.SUCCESS
    
    # Error info
    error_code: ErrorCode = ErrorCode.UNKNOWN
    error_message: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    latency_ms: float = 0.0
    first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Raw data
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    # Streaming
    is_streaming: bool = False
    is_complete: bool = True
    chunk_index: int = 0
    
    @property
    def has_reasoning(self) -> bool:
        """Check if response includes reasoning"""
        return bool(self.reasoning)
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if rate limited"""
        return self.error_code == ErrorCode.RATE_LIMIT
    
    @property
    def is_retryable(self) -> bool:
        """Check if error is retryable"""
        return self.error_code in (
            ErrorCode.RATE_LIMIT,
            ErrorCode.SERVER_ERROR,
            ErrorCode.TIMEOUT,
            ErrorCode.NETWORK_ERROR,
        )


@dataclass
class StreamChunk:
    """A chunk from a streaming response"""
    content: str = ""
    reasoning: str = ""
    finish_reason: str = ""
    delta_index: int = 0
    is_final: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing operation"""
    success: bool
    response: Optional[ParsedResponse] = None
    error: str = ""
    bytes_processed: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorDetector:
    """
    Detects and classifies API errors.
    
    Supports multiple API formats (OpenRouter, OpenAI, etc.)
    """
    
    # Error patterns by type
    ERROR_PATTERNS = {
        ErrorCode.INVALID_API_KEY: [
            'invalid api key',
            'invalid api_key',
            'authentication failed',
            'invalid authentication',
            'unauthorized',
            'api key not found',
            'incorrect api key',
        ],
        ErrorCode.RATE_LIMIT: [
            'rate limit',
            'too many requests',
            '429',
            'requests per minute',
            'quota exceeded',
            'limit exceeded',
            'throttl',
        ],
        ErrorCode.CONTEXT_TOO_LONG: [
            'context length exceeded',
            'maximum context length',
            'token limit exceeded',
            'context too long',
            'too many tokens',
        ],
        ErrorCode.MODEL_NOT_FOUND: [
            'model not found',
            'model does not exist',
            'unknown model',
            'model unavailable',
        ],
        ErrorCode.CONTENT_FILTERED: [
            'content filtered',
            'content policy',
            'safety',
            'inappropriate content',
            'blocked',
        ],
        ErrorCode.SERVER_ERROR: [
            'internal server error',
            '500',
            '502',
            '503',
            '504',
            'server error',
            'service unavailable',
            'bad gateway',
            'gateway timeout',
        ],
        ErrorCode.TIMEOUT: [
            'timeout',
            'timed out',
            'request timeout',
        ],
        ErrorCode.QUOTA_EXCEEDED: [
            'quota exceeded',
            'billing',
            'insufficient funds',
            'credit',
            'payment required',
        ],
    }
    
    @classmethod
    def detect_error(cls, response: Dict[str, Any]) -> tuple:
        """
        Detect error from response.
        
        Args:
            response: Raw API response
            
        Returns:
            Tuple of (ErrorCode, error_message)
        """
        # Check for explicit error
        if 'error' in response:
            error = response['error']
            
            if isinstance(error, str):
                return cls._classify_error(error), error
            
            if isinstance(error, dict):
                message = error.get('message', str(error))
                code = error.get('code', '')
                error_type = error.get('type', '')
                
                # Check code first
                if code:
                    code_lower = str(code).lower()
                    for err_code, patterns in cls.ERROR_PATTERNS.items():
                        if code_lower in patterns:
                            return err_code, message
                
                # Check message
                return cls._classify_error(message), message
        
        # Check HTTP status
        status = response.get('status', response.get('status_code', 0))
        if status >= 400:
            if status == 429:
                return ErrorCode.RATE_LIMIT, "Rate limit exceeded"
            elif status == 401:
                return ErrorCode.INVALID_API_KEY, "Authentication failed"
            elif status >= 500:
                return ErrorCode.SERVER_ERROR, f"Server error: {status}"
            elif status == 402:
                return ErrorCode.QUOTA_EXCEEDED, "Payment required"
        
        # No error detected
        return ErrorCode.UNKNOWN, ""
    
    @classmethod
    def _classify_error(cls, message: str) -> ErrorCode:
        """Classify error from message text"""
        message_lower = message.lower()
        
        for error_code, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in message_lower:
                    return error_code
        
        return ErrorCode.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingParser:
    """
    Parser for Server-Sent Events (SSE) streaming responses.
    
    Handles chunked responses efficiently for 4GB RAM device.
    
    Memory Strategy:
    - Process chunks as they arrive
    - Don't accumulate raw data
    - Use generators for streaming
    
    Usage:
        parser = StreamingParser()
        
        # From raw SSE stream
        for chunk in parser.parse_stream(raw_stream):
            print(chunk.content)
        
        # From lines
        parser.feed_line(line)
        if parser.has_chunk():
            chunk = parser.get_chunk()
    """
    
    # SSE markers
    DATA_PREFIX = "data: "
    DONE_MARKER = "[DONE]"
    
    def __init__(self, max_buffer_size: int = 1000000):
        """
        Initialize streaming parser.
        
        Args:
            max_buffer_size: Maximum buffer size in bytes (1MB default)
        """
        self._buffer = StringIO()
        self._max_buffer_size = max_buffer_size
        self._chunks: deque = deque(maxlen=100)  # Keep last 100 chunks
        self._lock = threading.Lock()
        
        # Current response state
        self._content_parts: List[str] = []
        self._reasoning_parts: List[str] = []
        self._total_chunks = 0
        self._start_time: Optional[float] = None
        self._first_token_time: Optional[float] = None
        
        # Statistics
        self._stats = {
            'total_chunks': 0,
            'total_bytes': 0,
            'total_content_chars': 0,
        }
    
    def parse_stream(
        self,
        stream: Generator[str, None, None]
    ) -> Generator[StreamChunk, None, None]:
        """
        Parse a streaming response.
        
        Args:
            stream: Generator yielding raw stream data
            
        Yields:
            StreamChunk objects
        """
        self._start_time = time.time()
        
        for data in stream:
            # Feed data
            chunks = self.feed_data(data)
            
            # Yield parsed chunks
            for chunk in chunks:
                yield chunk
    
    def feed_data(self, data: str) -> List[StreamChunk]:
        """
        Feed raw data to parser.
        
        Args:
            data: Raw stream data (may contain multiple lines)
            
        Returns:
            List of parsed chunks
        """
        chunks = []
        
        # Split by lines
        lines = data.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            chunk = self.feed_line(line)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def feed_line(self, line: str) -> Optional[StreamChunk]:
        """
        Feed a single line to parser.
        
        Args:
            line: Single line from stream
            
        Returns:
            StreamChunk if line contained complete data
        """
        # Skip empty lines
        if not line:
            return None
        
        # Check for data prefix
        if not line.startswith(self.DATA_PREFIX):
            return None
        
        # Extract data
        data_str = line[len(self.DATA_PREFIX):]
        
        # Check for done marker
        if data_str == self.DONE_MARKER:
            return StreamChunk(is_final=True)
        
        # Parse JSON
        try:
            data = json.loads(data_str)
            return self._parse_chunk(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse stream chunk: {e}")
            return None
    
    def _parse_chunk(self, data: Dict[str, Any]) -> Optional[StreamChunk]:
        """Parse a single chunk from JSON data"""
        self._total_chunks += 1
        self._stats['total_chunks'] += 1
        self._stats['total_bytes'] += len(json.dumps(data))
        
        # Initialize timing
        if self._first_token_time is None:
            self._first_token_time = time.time()
        
        # Extract choice (OpenAI/OpenRouter format)
        choices = data.get('choices', [])
        if not choices:
            return None
        
        choice = choices[0]
        delta = choice.get('delta', {})
        
        # Extract content
        content = delta.get('content', '')
        reasoning = delta.get('reasoning', '')
        
        # Extract finish reason
        finish_reason = choice.get('finish_reason', '')
        
        # Track content
        if content:
            self._content_parts.append(content)
            self._stats['total_content_chars'] += len(content)
        
        if reasoning:
            self._reasoning_parts.append(reasoning)
        
        chunk = StreamChunk(
            content=content,
            reasoning=reasoning,
            finish_reason=finish_reason,
            delta_index=self._total_chunks - 1,
            is_final=bool(finish_reason),
            raw_data=data,
        )
        
        self._chunks.append(chunk)
        
        return chunk
    
    def get_complete_response(self) -> ParsedResponse:
        """Get the complete response from accumulated chunks"""
        return ParsedResponse(
            content=''.join(self._content_parts),
            reasoning=''.join(self._reasoning_parts),
            success=True,
            response_type=ResponseType.STREAMING,
            is_streaming=True,
            is_complete=True,
            first_token_ms=(self._first_token_time - self._start_time) * 1000 if self._first_token_time else 0,
            latency_ms=(time.time() - self._start_time) * 1000 if self._start_time else 0,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics"""
        return {
            **self._stats,
            'content_length': len(''.join(self._content_parts)),
            'reasoning_length': len(''.join(self._reasoning_parts)),
        }
    
    def reset(self):
        """Reset parser for new stream"""
        self._buffer = StringIO()
        self._content_parts.clear()
        self._reasoning_parts.clear()
        self._chunks.clear()
        self._total_chunks = 0
        self._start_time = None
        self._first_token_time = None


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class ResponseParser:
    """
    Unified API Response Parser.
    
    Handles both regular and streaming responses from various APIs.
    Normalizes responses to a common format.
    
    Supported APIs:
    - OpenRouter
    - OpenAI
    - DeepSeek
    - Google (Gemini format)
    
    Memory Budget: < 1MB for streaming
    
    Usage:
        parser = ResponseParser()
        
        # Parse regular response
        response = parser.parse(json_response)
        if response.success:
            print(response.content)
        
        # Parse streaming response
        for chunk in parser.parse_stream(raw_stream):
            print(chunk.content, end='', flush=True)
    """
    
    def __init__(self):
        """Initialize response parser"""
        self._streaming_parser = StreamingParser()
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_parsed': 0,
            'successful': 0,
            'failed': 0,
            'streamed': 0,
            'total_tokens': 0,
        }
    
    def parse(
        self,
        response: Union[str, Dict[str, Any], bytes],
        model: str = "",
        latency_ms: float = 0.0
    ) -> ParsedResponse:
        """
        Parse an API response.
        
        Args:
            response: Raw API response (JSON string, dict, or bytes)
            model: Model that was used
            latency_ms: Request latency
            
        Returns:
            ParsedResponse with extracted data
        """
        with self._lock:
            self._stats['total_parsed'] += 1
            start_time = time.time()
        
        # Handle different input types
        if isinstance(response, bytes):
            try:
                response = response.decode('utf-8')
            except UnicodeDecodeError:
                return ParsedResponse(
                    success=False,
                    response_type=ResponseType.ERROR,
                    error_code=ErrorCode.NETWORK_ERROR,
                    error_message="Failed to decode response",
                )
        
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                return ParsedResponse(
                    success=False,
                    response_type=ResponseType.ERROR,
                    error_code=ErrorCode.UNKNOWN,
                    error_message=f"Invalid JSON: {e}",
                )
        
        # Now response should be a dict
        if not isinstance(response, dict):
            return ParsedResponse(
                success=False,
                response_type=ResponseType.ERROR,
                error_message="Invalid response format",
            )
        
        # Check for error
        error_code, error_message = ErrorDetector.detect_error(response)
        if error_code != ErrorCode.UNKNOWN:
            with self._lock:
                self._stats['failed'] += 1
            
            return ParsedResponse(
                success=False,
                response_type=ResponseType.ERROR,
                error_code=error_code,
                error_message=error_message,
                raw_response=response,
            )
        
        # Parse successful response
        parsed = self._parse_success(response, model, latency_ms)
        
        with self._lock:
            if parsed.success:
                self._stats['successful'] += 1
                self._stats['total_tokens'] += parsed.total_tokens
        
        return parsed
    
    def _parse_success(
        self,
        response: Dict[str, Any],
        model: str,
        latency_ms: float
    ) -> ParsedResponse:
        """Parse a successful response"""
        result = ParsedResponse(
            success=True,
            response_type=ResponseType.SUCCESS,
            model=model,
            latency_ms=latency_ms,
            raw_response=response,
        )
        
        # Extract from OpenAI/OpenRouter format
        choices = response.get('choices', [])
        if choices:
            choice = choices[0]
            
            # Get message
            message = choice.get('message', {})
            
            # Content
            result.content = message.get('content', '')
            
            # Reasoning (DeepSeek R1 style)
            result.reasoning = message.get('reasoning', '')
            
            # Finish reason
            result.finish_reason = choice.get('finish_reason', '')
        
        # Extract usage
        usage = response.get('usage', {})
        result.prompt_tokens = usage.get('prompt_tokens', 0)
        result.completion_tokens = usage.get('completion_tokens', 0)
        result.total_tokens = usage.get('total_tokens', 0)
        
        # Extract model info
        if not result.model:
            result.model = response.get('model', model)
        
        # Provider info
        result.provider = response.get('provider', '')
        
        # Calculate tokens per second
        if result.completion_tokens > 0 and latency_ms > 0:
            result.tokens_per_second = result.completion_tokens / (latency_ms / 1000)
        
        return result
    
    def parse_stream(
        self,
        stream: Generator[str, None, None],
        model: str = "",
        on_chunk: Callable[[StreamChunk], None] = None
    ) -> Generator[StreamChunk, None, ParsedResponse]:
        """
        Parse a streaming response.
        
        Args:
            stream: Raw stream generator
            model: Model being used
            on_chunk: Optional callback for each chunk
            
        Yields:
            StreamChunk objects
            
        Returns:
            Complete ParsedResponse when done
        """
        with self._lock:
            self._stats['total_parsed'] += 1
            self._stats['streamed'] += 1
        
        self._streaming_parser.reset()
        
        for chunk in self._streaming_parser.parse_stream(stream):
            if on_chunk:
                on_chunk(chunk)
            yield chunk
        
        # Get complete response
        result = self._streaming_parser.get_complete_response()
        result.model = model
        
        with self._lock:
            self._stats['successful'] += 1
        
        return result
    
    def parse_chunk(self, data: str) -> List[StreamChunk]:
        """Parse streaming chunk data"""
        return self._streaming_parser.feed_data(data)
    
    def extract_retry_after(self, response: Dict[str, Any]) -> Optional[float]:
        """
        Extract retry-after value from response.
        
        Args:
            response: Raw API response
            
        Returns:
            Retry-after seconds or None
        """
        # Check headers (if available)
        headers = response.get('headers', {})
        if 'retry-after' in headers:
            try:
                return float(headers['retry-after'])
            except ValueError:
                pass
        
        # Check error message
        error = response.get('error', {})
        if isinstance(error, dict):
            message = error.get('message', '')
            
            # Try to extract time from message
            # Common patterns: "try again in X seconds", "wait X ms"
            import re
            
            # Look for seconds
            match = re.search(r'(\d+(?:\.\d+)?)\s*seconds?', message, re.I)
            if match:
                return float(match.group(1))
            
            # Look for milliseconds
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)', message, re.I)
            if match:
                return float(match.group(1)) / 1000
            
            # Look for minutes
            match = re.search(r'(\d+(?:\.\d+)?)\s*minutes?', message, re.I)
            if match:
                return float(match.group(1)) * 60
        
        # Default for rate limit
        return 60.0  # 1 minute default
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parser statistics"""
        with self._lock:
            stats = self._stats.copy()
            
            if stats['total_parsed'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total_parsed']
            else:
                stats['success_rate'] = 0
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKED RESPONSE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class ChunkedResponseHandler:
    """
    Handler for chunked HTTP responses.
    
    Works with HTTP client for efficient streaming.
    """
    
    def __init__(self, chunk_size: int = 8192):
        """
        Initialize chunked handler.
        
        Args:
            chunk_size: Size of chunks to read
        """
        self._chunk_size = chunk_size
        self._buffer = ""
    
    def read_chunked(
        self,
        response,  # HTTP response object
        callback: Callable[[bytes], None] = None
    ) -> bytes:
        """
        Read chunked response.
        
        Args:
            response: HTTP response with read method
            callback: Optional callback for each chunk
            
        Returns:
            Complete response bytes
        """
        chunks = []
        
        while True:
            chunk = response.read(self._chunk_size)
            if not chunk:
                break
            
            chunks.append(chunk)
            
            if callback:
                callback(chunk)
        
        return b''.join(chunks)
    
    def iter_chunks(
        self,
        response,
        decode: bool = True
    ) -> Generator[Union[str, bytes], None, None]:
        """
        Iterate over chunks.
        
        Args:
            response: HTTP response with read method
            decode: Whether to decode as UTF-8
            
        Yields:
            Chunks as strings (if decode) or bytes
        """
        while True:
            chunk = response.read(self._chunk_size)
            if not chunk:
                break
            
            if decode:
                try:
                    yield chunk.decode('utf-8')
                except UnicodeDecodeError:
                    yield chunk.decode('utf-8', errors='replace')
            else:
                yield chunk


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_parser: Optional[ResponseParser] = None
_parser_lock = threading.Lock()  # FIX: Thread-safe singleton


def get_parser() -> ResponseParser:
    """Get global response parser instance (thread-safe)"""
    global _parser
    if _parser is None:
        with _parser_lock:
            if _parser is None:  # FIX: Double-check pattern prevents race condition
                _parser = ResponseParser()
    return _parser


def parse_response(response: Union[str, Dict], **kwargs) -> ParsedResponse:
    """Convenience function to parse response"""
    return get_parser().parse(response, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test() -> Dict[str, Any]:
    """Run self-test for response parser"""
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
    }
    
    parser = ResponseParser()
    
    # Test 1: Parse success response
    success_response = {
        "choices": [
            {
                "message": {
                    "content": "Hello, I'm JARVIS!",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        },
        "model": "test-model"
    }
    
    parsed = parser.parse(success_response)
    if parsed.success and parsed.content == "Hello, I'm JARVIS!":
        results['passed'].append('parse_success')
    else:
        results['failed'].append(f'parse_success: {parsed.error_message}')
    
    # Test 2: Parse error response
    error_response = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error"
        }
    }
    
    parsed = parser.parse(error_response)
    if not parsed.success and parsed.error_code == ErrorCode.RATE_LIMIT:
        results['passed'].append('parse_error')
    else:
        results['failed'].append(f'parse_error: {parsed.error_code}')
    
    # Test 3: Parse JSON string
    json_str = json.dumps(success_response)
    parsed = parser.parse(json_str)
    if parsed.success:
        results['passed'].append('parse_json_string')
    else:
        results['failed'].append('parse_json_string')
    
    # Test 4: Parse with reasoning (DeepSeek style)
    reasoning_response = {
        "choices": [
            {
                "message": {
                    "content": "The answer is 42",
                    "reasoning": "Let me think about this..."
                },
                "finish_reason": "stop"
            }
        ]
    }
    
    parsed = parser.parse(reasoning_response)
    if parsed.has_reasoning and "think about this" in parsed.reasoning:
        results['passed'].append('parse_reasoning')
    else:
        results['failed'].append('parse_reasoning')
    
    # Test 5: Error detection
    error_code, _ = ErrorDetector.detect_error({
        "error": {"message": "Invalid API key provided"}
    })
    if error_code == ErrorCode.INVALID_API_KEY:
        results['passed'].append('detect_auth_error')
    else:
        results['failed'].append('detect_auth_error')
    
    # Test 6: Streaming parser
    stream_parser = StreamingParser()
    
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: [DONE]',
    ]
    
    chunks = []
    for line in sse_lines:
        chunk = stream_parser.feed_line(line)
        if chunk:
            chunks.append(chunk)
    
    if len(chunks) >= 2 and chunks[-1].is_final:
        results['passed'].append('streaming_parser')
    else:
        results['failed'].append(f'streaming_parser: {len(chunks)} chunks')
    
    # Test 7: Complete response from stream
    complete = stream_parser.get_complete_response()
    if "Hello world" in complete.content:
        results['passed'].append('stream_complete')
    else:
        results['failed'].append(f'stream_complete: "{complete.content}"')
    
    # Test 8: Extract retry-after
    retry_response = {
        "error": {
            "message": "Rate limit exceeded. Try again in 30 seconds."
        }
    }
    retry_after = parser.extract_retry_after(retry_response)
    if retry_after == 30.0:
        results['passed'].append('retry_after_extraction')
    else:
        results['warnings'].append(f'retry_after_extraction: {retry_after}')
    
    results['stats'] = parser.get_stats()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Response Parser - Self Test")
    print("=" * 70)
    print(f"Device: Realme 2 Pro Lite (RMP2402)")
    print("-" * 70)
    
    test_results = self_test()
    
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
    
    print("\n📊 Statistics:")
    stats = test_results['stats']
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
