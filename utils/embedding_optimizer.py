#!/usr/bin/env python3
"""
Embedding Optimization Utilities
Provides batch processing and cost control for OpenAI embeddings
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingCostEstimate:
    """Cost estimation for embedding operations"""
    total_texts: int
    total_tokens: int
    estimated_cost: float
    batch_count: int
    cache_hits: int
    cache_savings: float

@dataclass
class EmbeddingCache:
    """Persistent cache for embeddings"""
    embeddings: Dict[str, List[float]]
    metadata: Dict[str, Dict]
    created_at: datetime
    last_accessed: datetime

class EmbeddingOptimizer:
    """Optimized embedding processor with batching and caching"""
    
    def __init__(self, openai_client, cache_dir: str = "embedding_cache"):
        self.openai_client = openai_client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # OpenAI API limits and pricing
        self.max_batch_size = 2048  # OpenAI embedding API limit
        self.cost_per_1k_tokens = 0.0001  # text-embedding-ada-002 pricing
        self.rate_limit_rpm = 3000  # requests per minute
        self.rate_limit_tpm = 1000000  # tokens per minute
        
        # Load persistent cache
        self.cache = self._load_cache()
        
        # Rate limiting tracking
        self.request_times = []
        self.token_usage = []
        
        logger.info(f"EmbeddingOptimizer initialized with cache at {cache_dir}")
    
    def _load_cache(self) -> EmbeddingCache:
        """Load persistent embedding cache"""
        cache_file = self.cache_dir / "embeddings.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(cache.embeddings)} entries")
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        # Return empty cache
        return EmbeddingCache(
            embeddings={},
            metadata={},
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        cache_file = self.cache_dir / "embeddings.pkl"
        
        try:
            self.cache.last_accessed = datetime.now()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _get_content_hash(self, text: str) -> str:
        """Generate hash for content to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def estimate_cost(self, texts: List[str]) -> EmbeddingCostEstimate:
        """Estimate cost and API usage for embedding operation"""
        cache_hits = 0
        total_tokens = 0
        unique_texts = []
        
        for text in texts:
            content_hash = self._get_content_hash(text)
            
            if content_hash in self.cache.embeddings:
                cache_hits += 1
            else:
                unique_texts.append(text)
                # Rough token estimate (1 token ≈ 4 characters for English)
                total_tokens += len(text) // 4
        
        texts_to_process = len(unique_texts)
        batch_count = (texts_to_process + self.max_batch_size - 1) // self.max_batch_size
        
        estimated_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
        cache_savings = (cache_hits / len(texts)) * estimated_cost if texts else 0
        
        return EmbeddingCostEstimate(
            total_texts=len(texts),
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            batch_count=batch_count,
            cache_hits=cache_hits,
            cache_savings=cache_savings
        )
    
    def _check_rate_limits(self, batch_size: int, estimated_tokens: int):
        """Check and enforce rate limits"""
        now = time.time()
        
        # Clean old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if now - t < 60]
        self.token_usage = [t for t in self.token_usage if t['time'] > now - 60]
        
        # Check RPM limit
        if len(self.request_times) >= self.rate_limit_rpm:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Check TPM limit
        recent_tokens = sum(t['tokens'] for t in self.token_usage)
        if recent_tokens + estimated_tokens > self.rate_limit_tpm:
            sleep_time = 60 - (now - self.token_usage[0]['time'])
            if sleep_time > 0:
                logger.warning(f"Token limit approaching, sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
    
    def get_embeddings_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Get embeddings for a batch of texts with optimization
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress information
            
        Returns:
            List of embeddings in the same order as input texts
        """
        if not texts:
            return []
        
        logger.info(f"Processing {len(texts)} texts for embeddings")
        
        # Get cost estimate
        cost_estimate = self.estimate_cost(texts)
        
        if show_progress:
            logger.info(f"Embedding cost estimate: ${cost_estimate.estimated_cost:.4f}")
            logger.info(f"Cache hits: {cost_estimate.cache_hits}/{cost_estimate.total_texts}")
            logger.info(f"Batches needed: {cost_estimate.batch_count}")
        
        # Collect results in original order
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache and prepare batch
        for i, text in enumerate(texts):
            content_hash = self._get_content_hash(text)
            
            if content_hash in self.cache.embeddings:
                embeddings.append(self.cache.embeddings[content_hash])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Process uncached texts in batches
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} new texts")
            
            new_embeddings = self._embed_texts_batched(texts_to_embed, show_progress)
            
            # Store in cache and fill placeholders
            for text, embedding, original_idx in zip(texts_to_embed, new_embeddings, text_indices):
                content_hash = self._get_content_hash(text)
                
                # Cache the embedding
                self.cache.embeddings[content_hash] = embedding
                self.cache.metadata[content_hash] = {
                    'created_at': datetime.now().isoformat(),
                    'text_length': len(text),
                    'model': 'text-embedding-ada-002'
                }
                
                # Fill placeholder
                embeddings[original_idx] = embedding
            
            # Save updated cache
            self._save_cache()
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def _embed_texts_batched(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """Embed texts using batch API calls"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_num = (i // self.max_batch_size) + 1
            total_batches = (len(texts) + self.max_batch_size - 1) // self.max_batch_size
            
            if show_progress:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Estimate tokens for rate limiting
            estimated_tokens = sum(len(text) // 4 for text in batch)
            
            # Check rate limits
            self._check_rate_limits(len(batch), estimated_tokens)
            
            try:
                # Make API call
                start_time = time.time()
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                # Track rate limiting
                self.request_times.append(time.time())
                self.token_usage.append({
                    'time': time.time(),
                    'tokens': estimated_tokens
                })
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                processing_time = time.time() - start_time
                if show_progress:
                    logger.info(f"Batch {batch_num} completed in {processing_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[0.0] * 1536 for _ in batch])
        
        return all_embeddings
    
    def cleanup_cache(self, max_age_days: int = 30, max_entries: int = 10000):
        """Clean up old cache entries"""
        if not self.cache.embeddings:
            return
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Remove old entries
        old_keys = []
        for key, metadata in self.cache.metadata.items():
            created_at = datetime.fromisoformat(metadata.get('created_at', '1970-01-01'))
            if created_at < cutoff_date:
                old_keys.append(key)
        
        for key in old_keys:
            del self.cache.embeddings[key]
            del self.cache.metadata[key]
        
        # Limit total entries (keep most recent)
        if len(self.cache.embeddings) > max_entries:
            # Sort by creation time, keep newest
            sorted_items = sorted(
                self.cache.metadata.items(),
                key=lambda x: x[1].get('created_at', '1970-01-01'),
                reverse=True
            )
            
            keys_to_keep = [key for key, _ in sorted_items[:max_entries]]
            keys_to_remove = set(self.cache.embeddings.keys()) - set(keys_to_keep)
            
            for key in keys_to_remove:
                del self.cache.embeddings[key]
                del self.cache.metadata[key]
        
        if old_keys or len(self.cache.embeddings) != len(self.cache.metadata):
            logger.info(f"Cache cleanup: removed {len(old_keys)} old entries")
            self._save_cache()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache.embeddings),
            'cache_size_mb': len(pickle.dumps(self.cache)) / (1024 * 1024),
            'created_at': self.cache.created_at.isoformat(),
            'last_accessed': self.cache.last_accessed.isoformat(),
            'hit_rate_estimate': 'unknown'  # Would need usage tracking
        }
