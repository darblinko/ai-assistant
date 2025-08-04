"""
Utility modules for multimodal RAG assistant
"""

from .embedding_optimizer import EmbeddingOptimizer, EmbeddingCostEstimate
from .optimized_processor import OptimizedMultimodalProcessor, ProcessingStats

__all__ = [
    'EmbeddingOptimizer',
    'EmbeddingCostEstimate',
    'OptimizedMultimodalProcessor',
    'ProcessingStats'
]
