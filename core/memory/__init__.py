#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Memory System Package
============================================

Memory management for JARVIS AI system.

Modules:
    - chat_storage: Persistent chat storage with SQLite
    - context_manager: Conversation context management
    - memory_optimizer: RAM optimization for 4GB device
    - conversation_indexer: Search and indexing

Exports:
    - ChatStorage: Persistent storage engine
    - ContextManager: Context window management
    - MemoryOptimizer: Memory optimization
    - ConversationIndexer: Search and indexing
"""

from .chat_storage import (
    ChatStorage,
    Message,
    Conversation,
    MessageRole,
    MessageStatus,
    ConversationStatus,
    SearchResult as StorageSearchResult,
    StorageStats,
    get_storage,
    initialize_storage,
)

from .context_manager import (
    ContextManager,
    ContextWindow,
    ContextMessage,
    ContextPriority,
    ContextSnapshot,
    SummarizationTrigger,
    SummarizationResult,
    TokenEstimator,
    get_context_manager,
    initialize_context_manager,
)

from .memory_optimizer import (
    MemoryOptimizer,
    MemoryLevel,
    MemoryStats,
    MemoryEvent,
    MemoryAwareLRUCache,
    LazyLoader,
    get_process_memory_mb,
    get_system_memory_mb,
    estimate_object_size,
    memory_efficient,
    memory_limited,
    get_memory_optimizer,
    initialize_memory_optimizer,
)

from .conversation_indexer import (
    ConversationIndexer,
    SearchQuery,
    SearchResult,
    SearchType,
    SortOrder,
    IndexedMessage,
    IndexStats,
    TextProcessor,
    get_indexer,
    initialize_indexer,
)

__all__ = [
    # Chat Storage
    'ChatStorage',
    'Message',
    'Conversation',
    'MessageRole',
    'MessageStatus',
    'ConversationStatus',
    'StorageSearchResult',
    'StorageStats',
    'get_storage',
    'initialize_storage',
    
    # Context Manager
    'ContextManager',
    'ContextWindow',
    'ContextMessage',
    'ContextPriority',
    'ContextSnapshot',
    'SummarizationTrigger',
    'SummarizationResult',
    'TokenEstimator',
    'get_context_manager',
    'initialize_context_manager',
    
    # Memory Optimizer
    'MemoryOptimizer',
    'MemoryLevel',
    'MemoryStats',
    'MemoryEvent',
    'MemoryAwareLRUCache',
    'LazyLoader',
    'get_process_memory_mb',
    'get_system_memory_mb',
    'estimate_object_size',
    'memory_efficient',
    'memory_limited',
    'get_memory_optimizer',
    'initialize_memory_optimizer',
    
    # Conversation Indexer
    'ConversationIndexer',
    'SearchQuery',
    'SearchResult',
    'SearchType',
    'SortOrder',
    'IndexedMessage',
    'IndexStats',
    'TextProcessor',
    'get_indexer',
    'initialize_indexer',
]
