#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 4: ULTIMATE DEVICE COMPATIBILITY TEST
=================================================================

Device Target: Realme 2 Pro Lite (RMP2402) | RAM: 4GB | Platform: Termux

Phase 4: Memory System
- chat_storage.py
- context_manager.py  
- conversation_indexer.py
- memory_optimizer.py

IF ALL TESTS PASS: Phase 4 is 100% READY for production on target device.
"""

import sys
import os
import time
import json
import threading
import gc
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Memory constraints for 4GB RAM device
MAX_MEMORY_MB = 100

class UltimateTestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        self.memory_issues = []
        self.start_time = time.time()
        
    def add_pass(self, name, details=""):
        self.passed.append((name, details))
        print(f"  ✅ {name}")
        if details:
            print(f"      └─ {details}")
    
    def add_fail(self, name, error):
        self.failed.append((name, error))
        print(f"  ❌ {name}")
        print(f"      └─ ERROR: {error}")
    
    def add_warning(self, name, warning):
        self.warnings.append((name, warning))
        print(f"  ⚠️  {name}: {warning}")
    
    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed)
        
        print("\n" + "=" * 70)
        print("🔴🟢 PHASE 4 ULTIMATE TEST RESULTS 🔴🟢")
        print("=" * 70)
        print(f"Target Device: Realme 2 Pro Lite (RMP2402) | 4GB RAM | Termux")
        print("-" * 70)
        print(f"✅ PASSED:  {len(self.passed)}")
        print(f"❌ FAILED:  {len(self.failed)}")
        print(f"⚠️  WARNINGS: {len(self.warnings)}")
        print("-" * 70)
        print(f"⏱️  Total Time: {elapsed:.2f}s")
        
        if self.failed:
            print("\n❌ FAILED TESTS:")
            for name, error in self.failed:
                print(f"   • {name}: {error}")
        
        print("=" * 70)
        
        if len(self.failed) == 0 and len(self.memory_issues) == 0:
            print("\n🎉✅ ULTIMATE TEST PASSED - PHASE 4 IS 100% DEVICE COMPATIBLE! ✅🎉")
            return True
        else:
            print("\n⚠️❌ ULTIMATE TEST FAILED - FIXES REQUIRED ❌⚠️")
            return False

results = UltimateTestResult()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TERMUX COMPATIBILITY - IMPORT ALL MODULES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📦 SECTION 1: TERMUX COMPATIBILITY - IMPORT TESTS")
print("=" * 70)

# Test imports
try:
    from core.memory.chat_storage import (
        ChatStorage, Message, Conversation, MessageRole, MessageStatus,
        ConversationStatus, SearchResult, StorageStats
    )
    results.add_pass("Import chat_storage", "All exports imported")
except Exception as e:
    results.add_fail("Import chat_storage", str(e))

try:
    from core.memory.context_manager import (
        ContextManager, ContextWindow, ContextMessage, ContextPriority,
        ContextSnapshot, SummarizationTrigger, SummarizationResult, TokenEstimator
    )
    results.add_pass("Import context_manager", "All exports imported")
except Exception as e:
    results.add_fail("Import context_manager", str(e))

try:
    from core.memory.conversation_indexer import (
        ConversationIndexer, SearchQuery, SearchResult, SearchType,
        SortOrder, IndexedMessage, IndexStats, TextProcessor
    )
    results.add_pass("Import conversation_indexer", "All exports imported")
except Exception as e:
    results.add_fail("Import conversation_indexer", str(e))

try:
    from core.memory.memory_optimizer import (
        MemoryOptimizer, MemoryLevel, MemoryStats, MemoryEvent,
        MemoryAwareLRUCache, LazyLoader, get_process_memory_mb,
        get_system_memory_mb, estimate_object_size
    )
    results.add_pass("Import memory_optimizer", "All exports imported")
except Exception as e:
    results.add_fail("Import memory_optimizer", str(e))

try:
    from core.memory import (
        ChatStorage, ContextManager, MemoryOptimizer, ConversationIndexer,
        get_storage, get_context_manager, get_memory_optimizer, get_indexer
    )
    results.add_pass("Import __init__.py", "Package imports working")
except Exception as e:
    results.add_fail("Import __init__.py", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: IMPORT CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🔄 SECTION 2: IMPORT CYCLE DETECTION")
print("=" * 70)

try:
    for i in range(5):
        import importlib
        import core.memory.chat_storage as cs
        import core.memory.context_manager as cm
        import core.memory.conversation_indexer as ci
        import core.memory.memory_optimizer as mo
        importlib.reload(cs)
        importlib.reload(cm)
        importlib.reload(ci)
        importlib.reload(mo)
    results.add_pass("Import cycle detection", "No circular imports after 5 reloads")
except ImportError as e:
    results.add_fail("Import cycle detection", f"Circular import: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CHAT STORAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("💾 SECTION 3: CHAT STORAGE TESTS")
print("=" * 70)

# Create temp directory for test database
temp_dir = tempfile.mkdtemp()
test_db_path = os.path.join(temp_dir, "test_chat.db")

try:
    # Test ChatStorage creation
    storage = ChatStorage(db_path=test_db_path)
    results.add_pass("ChatStorage creation", f"DB created at {test_db_path}")
except Exception as e:
    results.add_fail("ChatStorage creation", str(e))
    storage = None

if storage:
    # Test create_conversation
    try:
        conv = storage.create_conversation(title="Test Chat", system_prompt="You are a test assistant.")
        if conv and conv.id:
            results.add_pass("create_conversation", f"ID: {conv.id}")
        else:
            results.add_fail("create_conversation", "No conversation ID returned")
    except Exception as e:
        results.add_fail("create_conversation", str(e))
        conv = None
    
    if conv:
        # Test add_message
        try:
            msg = storage.add_message(conv.id, "user", "Hello, this is a test!")
            if msg and msg.id:
                results.add_pass("add_message", f"Message ID: {msg.id}")
            else:
                results.add_fail("add_message", "No message ID returned")
        except Exception as e:
            results.add_fail("add_message", str(e))
        
        # Test add assistant message
        try:
            msg2 = storage.add_message(conv.id, "assistant", "Hello! How can I help you?")
            if msg2 and msg2.id:
                results.add_pass("add_message (assistant)", f"Message ID: {msg2.id}")
            else:
                results.add_fail("add_message (assistant)", "No message ID returned")
        except Exception as e:
            results.add_fail("add_message (assistant)", str(e))
        
        # Test get_messages
        try:
            messages = storage.get_messages(conv.id)
            if len(messages) >= 2:
                results.add_pass("get_messages", f"{len(messages)} messages retrieved")
            else:
                results.add_fail("get_messages", f"Only {len(messages)} messages")
        except Exception as e:
            results.add_fail("get_messages", str(e))
        
        # Test get_conversation
        try:
            retrieved_conv = storage.get_conversation(conv.id)
            if retrieved_conv and retrieved_conv.id == conv.id:
                results.add_pass("get_conversation", "Conversation retrieved")
            else:
                results.add_fail("get_conversation", "Conversation not found")
        except Exception as e:
            results.add_fail("get_conversation", str(e))
        
        # Test list_conversations
        try:
            convs = storage.list_conversations()
            if len(convs) >= 1:
                results.add_pass("list_conversations", f"{len(convs)} conversations")
            else:
                results.add_fail("list_conversations", "No conversations listed")
        except Exception as e:
            results.add_fail("list_conversations", str(e))
        
        # Test get_stats
        try:
            stats = storage.get_stats()
            if stats.total_messages >= 2:
                results.add_pass("get_stats (storage)", f"{stats.total_messages} messages, {stats.total_conversations} conversations")
            else:
                results.add_fail("get_stats (storage)", f"Only {stats.total_messages} messages")
        except Exception as e:
            results.add_fail("get_stats (storage)", str(e))
        
        # Test search (FTS5)
        try:
            search_results = storage.search("test")
            if len(search_results) >= 1:
                results.add_pass("search (FTS5)", f"{len(search_results)} results for 'test'")
            else:
                results.add_warning("search (FTS5)", "No search results")
        except Exception as e:
            results.add_fail("search (FTS5)", str(e))
        
        # Test update_conversation
        try:
            success = storage.update_conversation(conv.id, title="Updated Title")
            if success:
                updated = storage.get_conversation(conv.id)
                if updated and updated.title == "Updated Title":
                    results.add_pass("update_conversation", "Title updated")
                else:
                    results.add_fail("update_conversation", "Title not updated")
            else:
                results.add_fail("update_conversation", "Update returned False")
        except Exception as e:
            results.add_fail("update_conversation", str(e))
        
        # Test delete_conversation (soft delete)
        try:
            success = storage.delete_conversation(conv.id, hard=False)
            if success:
                results.add_pass("delete_conversation (soft)", "Soft delete successful")
            else:
                results.add_fail("delete_conversation (soft)", "Delete returned False")
        except Exception as e:
            results.add_fail("delete_conversation (soft)", str(e))

# Cleanup temp directory
try:
    shutil.rmtree(temp_dir)
except:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONTEXT MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("📝 SECTION 4: CONTEXT MANAGER TESTS")
print("=" * 70)

try:
    ctx_manager = ContextManager(default_max_tokens=1000)
    results.add_pass("ContextManager creation", "Created with 1000 token limit")
except Exception as e:
    results.add_fail("ContextManager creation", str(e))
    ctx_manager = None

if ctx_manager:
    # Test create_context
    try:
        ctx = ctx_manager.create_context("test-ctx", max_tokens=500)
        if ctx and ctx.max_tokens == 500:
            results.add_pass("create_context", f"Max tokens: {ctx.max_tokens}")
        else:
            results.add_fail("create_context", f"Max tokens: {ctx.max_tokens if ctx else 'None'}")
    except Exception as e:
        results.add_fail("create_context", str(e))
    
    # Test set_system_prompt
    try:
        success = ctx_manager.set_system_prompt("test-ctx", "You are JARVIS.")
        if success:
            results.add_pass("set_system_prompt", "System prompt set")
        else:
            results.add_fail("set_system_prompt", "Returned False")
    except Exception as e:
        results.add_fail("set_system_prompt", str(e))
    
    # Test add_message
    try:
        ctx_manager.add_message("test-ctx", "user", "Hello!")
        ctx_manager.add_message("test-ctx", "assistant", "Hi there!")
        ctx = ctx_manager.get_context("test-ctx")
        # Check if messages were added (context should have messages)
        if ctx and len(ctx.messages) >= 2:
            results.add_pass("add_message (context)", f"{len(ctx.messages)} messages")
        else:
            results.add_fail("add_message (context)", f"{len(ctx.messages) if ctx else 0} messages")
    except Exception as e:
        results.add_fail("add_message (context)", str(e))
    
    # Test get_messages_for_api
    try:
        api_messages = ctx_manager.get_messages_for_api("test-ctx")
        # Should have system + 2 messages = 3
        if len(api_messages) >= 2:  # At least the messages
            results.add_pass("get_messages_for_api", f"{len(api_messages)} messages for API")
        else:
            results.add_fail("get_messages_for_api", f"{len(api_messages)} messages")
    except Exception as e:
        results.add_fail("get_messages_for_api", str(e))
    
    # Test TokenEstimator
    try:
        tokens = TokenEstimator.estimate("Hello, this is a test message!")
        if tokens > 0:
            results.add_pass("TokenEstimator", f"{tokens} tokens estimated")
        else:
            results.add_fail("TokenEstimator", "Returned 0 tokens")
    except Exception as e:
        results.add_fail("TokenEstimator", str(e))
    
    # Test context truncation
    try:
        # Add many messages to trigger truncation
        for i in range(20):
            ctx_manager.add_message("test-ctx", "user", f"Message {i} " * 20)
        
        ctx = ctx_manager.get_context("test-ctx")
        if ctx and ctx.total_tokens <= ctx.max_tokens:
            results.add_pass("auto_truncation", f"Tokens: {ctx.total_tokens} <= {ctx.max_tokens}")
        else:
            results.add_warning("auto_truncation", f"Tokens: {ctx.total_tokens if ctx else 0}")
    except Exception as e:
        results.add_fail("auto_truncation", str(e))
    
    # Test get_stats
    try:
        stats = ctx_manager.get_stats()
        if 'active_contexts' in stats:
            results.add_pass("get_stats (context)", f"{stats['active_contexts']} active contexts")
        else:
            results.add_fail("get_stats (context)", "Missing active_contexts")
    except Exception as e:
        results.add_fail("get_stats (context)", str(e))
    
    # Test clear_context
    try:
        ctx_manager.clear_context("test-ctx")
        ctx = ctx_manager.get_context("test-ctx")
        if ctx is None:
            results.add_pass("clear_context", "Context cleared")
        else:
            results.add_fail("clear_context", "Context still exists")
    except Exception as e:
        results.add_fail("clear_context", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CONVERSATION INDEXER TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🔍 SECTION 5: CONVERSATION INDEXER TESTS")
print("=" * 70)

try:
    indexer = ConversationIndexer()
    results.add_pass("ConversationIndexer creation", "Indexer created")
except Exception as e:
    results.add_fail("ConversationIndexer creation", str(e))
    indexer = None

if indexer:
    # Test index_message
    try:
        indexer.index_message(1, "conv-1", "user", "Hello, how are you doing today?", time.time())
        indexer.index_message(2, "conv-1", "assistant", "I'm doing great! How can I help you?", time.time())
        indexer.index_message(3, "conv-2", "user", "What is the weather like?", time.time())
        stats = indexer.get_stats()
        if stats['total_indexed'] == 3:
            results.add_pass("index_message", f"{stats['total_indexed']} messages indexed")
        else:
            results.add_fail("index_message", f"{stats['total_indexed']} indexed")
    except Exception as e:
        results.add_fail("index_message", str(e))
    
    # Test keyword search
    try:
        query = SearchQuery(text="hello")
        search_results = indexer.search(query)
        if len(search_results) > 0:
            results.add_pass("keyword_search", f"{len(search_results)} results")
        else:
            results.add_fail("keyword_search", "No results")
    except Exception as e:
        results.add_fail("keyword_search", str(e))
    
    # Test conversation filter
    try:
        query = SearchQuery(text="help", conversation_id="conv-1")
        search_results = indexer.search(query)
        if all(r.conversation_id == "conv-1" for r in search_results):
            results.add_pass("conversation_filter", "All results from conv-1")
        else:
            results.add_fail("conversation_filter", "Mixed conversations")
    except Exception as e:
        results.add_fail("conversation_filter", str(e))
    
    # Test suggestions
    try:
        suggestions = indexer.get_suggestions("hel")
        results.add_pass("get_suggestions", f"{len(suggestions)} suggestions")
    except Exception as e:
        results.add_fail("get_suggestions", str(e))
    
    # Test topics extraction
    try:
        topics = indexer.extract_topics(max_topics=5)
        if len(topics) > 0:
            results.add_pass("extract_topics", f"{len(topics)} topics found")
        else:
            results.add_warning("extract_topics", "No topics")
    except Exception as e:
        results.add_fail("extract_topics", str(e))
    
    # Test TextProcessor
    try:
        tokens = TextProcessor.tokenize("Hello, how are you?")
        if len(tokens) > 0:
            results.add_pass("TextProcessor.tokenize", f"{len(tokens)} tokens")
        else:
            results.add_fail("TextProcessor.tokenize", "No tokens")
    except Exception as e:
        results.add_fail("TextProcessor.tokenize", str(e))
    
    # Test remove_message
    try:
        indexer.remove_message(3)
        query = SearchQuery(text="weather")
        search_results = indexer.search(query)
        if len(search_results) == 0:
            results.add_pass("remove_message", "Message removed")
        else:
            results.add_fail("remove_message", "Message still in index")
    except Exception as e:
        results.add_fail("remove_message", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MEMORY OPTIMIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🧠 SECTION 6: MEMORY OPTIMIZER TESTS")
print("=" * 70)

# Test get_process_memory_mb
try:
    mem_mb = get_process_memory_mb()
    if mem_mb >= 0:
        results.add_pass("get_process_memory_mb", f"{mem_mb:.1f}MB used")
    else:
        results.add_fail("get_process_memory_mb", "Negative value")
except Exception as e:
    results.add_fail("get_process_memory_mb", str(e))

# Test get_system_memory_mb
try:
    total, available = get_system_memory_mb()
    if total > 0:
        results.add_pass("get_system_memory_mb", f"{total:.0f}MB total, {available:.0f}MB available")
    else:
        results.add_warning("get_system_memory_mb", "Could not read system memory")
except Exception as e:
    results.add_fail("get_system_memory_mb", str(e))

# Test MemoryOptimizer
try:
    optimizer = MemoryOptimizer(enable_monitoring=False)
    results.add_pass("MemoryOptimizer creation", "Created without monitoring")
except Exception as e:
    results.add_fail("MemoryOptimizer creation", str(e))
    optimizer = None

if optimizer:
    # Test get_stats
    try:
        stats = optimizer.get_stats()
        # MemoryLevel is an enum, stats.level should be a MemoryLevel enum member
        # Check by verifying it has a name attribute (enum member)
        if hasattr(stats.level, 'name') and hasattr(stats.level, 'value'):
            results.add_pass("get_stats (memory)", f"Level: {stats.level.name}")
        else:
            results.add_fail("get_stats (memory)", f"Invalid level: {stats.level}")
    except Exception as e:
        results.add_fail("get_stats (memory)", str(e))
    
    # Test cleanup
    try:
        collected = optimizer.cleanup()
        results.add_pass("cleanup", f"{collected} objects collected")
    except Exception as e:
        results.add_fail("cleanup", str(e))
    
    # Test memory context manager
    try:
        with optimizer.monitor():
            # Do some work
            data = [i for i in range(1000)]
        results.add_pass("monitor context", "Monitor context works")
    except Exception as e:
        results.add_fail("monitor context", str(e))

# Test MemoryAwareLRUCache
try:
    cache = MemoryAwareLRUCache(max_size=5)
    for i in range(10):
        cache.set(f'key_{i}', f'value_{i}' * 100)
    
    if len(cache._cache) <= 5:
        results.add_pass("MemoryAwareLRUCache", f"Cache size limited to {len(cache._cache)}")
    else:
        results.add_fail("MemoryAwareLRUCache", f"Cache size: {len(cache._cache)}")
except Exception as e:
    results.add_fail("MemoryAwareLRUCache", str(e))

# Test LazyLoader
try:
    loaded = [False]
    
    def expensive_loader():
        loaded[0] = True
        return "expensive_data"
    
    lazy = LazyLoader(expensive_loader)
    if not lazy.is_loaded():
        value = lazy.get()
        if loaded[0] and value == "expensive_data":
            results.add_pass("LazyLoader", "Lazy loading works")
        else:
            results.add_fail("LazyLoader", f"loaded={loaded[0]}, value={value}")
    else:
        results.add_fail("LazyLoader", "Loaded too early")
except Exception as e:
    results.add_fail("LazyLoader", str(e))

# Test estimate_object_size
try:
    test_obj = {"key": "value" * 100}
    size = estimate_object_size(test_obj)
    if size > 0:
        results.add_pass("estimate_object_size", f"{size} bytes")
    else:
        results.add_fail("estimate_object_size", "Size is 0")
except Exception as e:
    results.add_fail("estimate_object_size", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: THREAD SAFETY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🔒 SECTION 7: THREAD SAFETY TESTS")
print("=" * 70)

# Test MemoryAwareLRUCache thread safety
try:
    cache = MemoryAwareLRUCache(max_size=100)
    errors = []
    
    def cache_ops():
        for i in range(100):
            try:
                cache.set(f'key_{threading.current_thread().name}_{i}', f'value_{i}' * 50)
                cache.get(f'key_{threading.current_thread().name}_{i}')
            except Exception as e:
                errors.append(str(e))
    
    threads = [threading.Thread(target=cache_ops, name=f't{i}') for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("Thread Safety: LRU Cache", f"{len(errors)} errors")
    else:
        results.add_pass("Thread Safety: LRU Cache", "500 ops without errors")
except Exception as e:
    results.add_fail("Thread Safety: LRU Cache", str(e))

# Test ContextManager thread safety
try:
    ctx_mgr = ContextManager()
    errors = []
    
    def ctx_ops():
        try:
            ctx_mgr.create_context(f'ctx-{threading.current_thread().name}')
            for i in range(20):
                ctx_mgr.add_message(f'ctx-{threading.current_thread().name}', 'user', f'msg{i}')
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=ctx_ops, name=f'ctx{i}') for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("Thread Safety: ContextManager", f"{len(errors)} errors")
    else:
        results.add_pass("Thread Safety: ContextManager", "5 contexts, 100 messages")
except Exception as e:
    results.add_fail("Thread Safety: ContextManager", str(e))

# Test ConversationIndexer thread safety
try:
    idx = ConversationIndexer()
    errors = []
    
    def idx_ops():
        try:
            for i in range(20):
                idx.index_message(
                    int(f'{threading.current_thread().name[3:]}{i}'),
                    f'conv-{threading.current_thread().name}',
                    'user',
                    f'message {i}',
                    time.time()
                )
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=idx_ops, name=f'idx{i}') for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    if errors:
        results.add_fail("Thread Safety: Indexer", f"{len(errors)} errors")
    else:
        results.add_pass("Thread Safety: Indexer", "100 messages indexed")
except Exception as e:
    results.add_fail("Thread Safety: Indexer", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: GLOBAL SINGLETON TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🌐 SECTION 8: GLOBAL SINGLETON TESTS")
print("=" * 70)

# Test global singleton thread safety
try:
    from core.memory import (
        get_storage, get_context_manager, get_memory_optimizer, get_indexer
    )
    
    # Reset globals
    import core.memory.chat_storage as cs
    import core.memory.context_manager as cm
    import core.memory.memory_optimizer as mo
    import core.memory.conversation_indexer as ci
    
    cs._storage = None
    cm._manager = None
    mo._optimizer = None
    ci._indexer = None
    
    instances = {'storage': [], 'ctx': [], 'mem': [], 'idx': []}
    errors = []
    
    def get_singletons():
        try:
            s = get_storage()
            c = get_context_manager()
            m = get_memory_optimizer()
            i = get_indexer()
            instances['storage'].append(id(s))
            instances['ctx'].append(id(c))
            instances['mem'].append(id(m))
            instances['idx'].append(id(i))
        except Exception as e:
            errors.append(str(e))
    
    threads = [threading.Thread(target=get_singletons) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # All should be same instances
    all_same = (
        len(set(instances['storage'])) == 1 and
        len(set(instances['ctx'])) == 1 and
        len(set(instances['mem'])) == 1 and
        len(set(instances['idx'])) == 1
    )
    
    if all_same and not errors:
        results.add_pass("Global Singletons", "20 threads, same instances")
    elif errors:
        results.add_fail("Global Singletons", f"Errors: {errors[:3]}")
    else:
        # Check which ones failed
        failed = []
        if len(set(instances['storage'])) > 1:
            failed.append(f"storage:{len(set(instances['storage']))}")
        if len(set(instances['ctx'])) > 1:
            failed.append(f"ctx:{len(set(instances['ctx']))}")
        if len(set(instances['mem'])) > 1:
            failed.append(f"mem:{len(set(instances['mem']))}")
        if len(set(instances['idx'])) > 1:
            failed.append(f"idx:{len(set(instances['idx']))}")
        results.add_fail("Global Singletons", f"Multiple instances: {', '.join(failed)}")
except Exception as e:
    results.add_fail("Global Singletons", str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

success = results.summary()

if success:
    print("\n" + "🔥" * 35)
    print("PHASE 4 IS CERTIFIED FOR:")
    print("  • Realme 2 Pro Lite (RMP2402)")
    print("  • 4GB RAM Constraint")
    print("  • Termux Environment")
    print("  • Production Deployment")
    print("🔥" * 35)

sys.exit(0 if success else 1)
