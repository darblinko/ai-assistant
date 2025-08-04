#!/usr/bin/env python3
"""
Optimized Multimodal Processor
Enhanced version with batch embedding processing and smart chunking
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import time
from datetime import datetime

from utils.embedding_optimizer import EmbeddingOptimizer, EmbeddingCostEstimate

logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Statistics for document processing"""
    total_elements: int
    processed_elements: int
    skipped_elements: int
    text_blocks: int
    tables: int
    figures: int
    embeddings_generated: int
    embeddings_cached: int
    processing_time: float
    cost_estimate: float

class OptimizedMultimodalProcessor:
    """Enhanced multimodal processor with batch optimization"""
    
    def __init__(self, openai_client, chroma_client, web_search_agent=None):
        self.openai_client = openai_client
        self.chroma_client = chroma_client
        self.web_search_agent = web_search_agent
        
        # Initialize embedding optimizer
        self.embedding_optimizer = EmbeddingOptimizer(openai_client)
        
        # Processing configuration
        self.min_text_length = int(os.getenv('MIN_TEXT_LENGTH', 30))
        self.max_text_length = int(os.getenv('MAX_TEXT_LENGTH', 2000))
        self.combine_small_blocks = os.getenv('COMBINE_SMALL_BLOCKS', 'true').lower() == 'true'
        self.skip_duplicate_content = os.getenv('SKIP_DUPLICATE_CONTENT', 'true').lower() == 'true'
        
        # Initialize collections
        self.collections = {}
        self._initialize_collections()
        
        logger.info("OptimizedMultimodalProcessor initialized with batch processing")
    
    def _initialize_collections(self):
        """Initialize collections with proper error handling"""
        collection_names = ["service_manual_text", "service_manual_tables", "service_manual_figures"]
        
        for name in collection_names:
            try:
                # Try to get existing collection first
                collection = self.chroma_client.get_collection(name)
                logger.info(f"Found existing collection: {name} with {collection.count()} items")
            except Exception:
                # Create new collection if it doesn't exist
                try:
                    collection = self.chroma_client.create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Created new collection: {name}")
                except Exception as e:
                    logger.error(f"Failed to create collection {name}: {e}")
                    collection = None
            
            self.collections[name] = collection
        
        # Set collection references for backward compatibility
        self.text_collection = self.collections.get("service_manual_text")
        self.table_collection = self.collections.get("service_manual_tables")
        self.figure_collection = self.collections.get("service_manual_figures")
    
    def process_and_store_elements_optimized(self, elements: List, manual_name: str) -> ProcessingStats:
        """
        Process and store elements with batch optimization
        
        Args:
            elements: List of DocumentElement objects
            manual_name: Name of the manual being processed
            
        Returns:
            ProcessingStats object with processing information
        """
        start_time = time.time()
        
        logger.info(f"Starting optimized processing of {len(elements)} elements for {manual_name}")
        
        # Step 1: Filter and optimize elements
        optimized_elements = self._optimize_elements(elements)
        
        # Step 2: Extract all text content for batch embedding
        texts_to_embed = []
        element_text_map = {}  # Map from text index to element
        
        for element in optimized_elements:
            if element.content and len(element.content.strip()) >= self.min_text_length:
                text_idx = len(texts_to_embed)
                texts_to_embed.append(element.content)
                element_text_map[text_idx] = element
        
        logger.info(f"Prepared {len(texts_to_embed)} texts for batch embedding")
        
        # Step 3: Get cost estimate
        cost_estimate = self.embedding_optimizer.estimate_cost(texts_to_embed)
        
        logger.info(f"Embedding cost estimate: ${cost_estimate.estimated_cost:.4f}")
        logger.info(f"Cache hits: {cost_estimate.cache_hits}/{cost_estimate.total_texts}")
        
        # Step 4: Batch generate embeddings
        if texts_to_embed:
            embeddings = self.embedding_optimizer.get_embeddings_batch(texts_to_embed)
            
            # Assign embeddings back to elements
            for text_idx, embedding in enumerate(embeddings):
                if text_idx in element_text_map:
                    element_text_map[text_idx].text_embedding = embedding
        
        # Step 5: Store elements in appropriate collections
        stored_counts = self._store_elements_batch(optimized_elements)
        
        # Step 6: Calculate processing statistics
        processing_time = time.time() - start_time
        
        stats = ProcessingStats(
            total_elements=len(elements),
            processed_elements=len(optimized_elements),
            skipped_elements=len(elements) - len(optimized_elements),
            text_blocks=stored_counts.get("text", 0),
            tables=stored_counts.get("tables", 0),
            figures=stored_counts.get("figures", 0),
            embeddings_generated=len(texts_to_embed) - cost_estimate.cache_hits,
            embeddings_cached=cost_estimate.cache_hits,
            processing_time=processing_time,
            cost_estimate=cost_estimate.estimated_cost
        )
        
        logger.info(f"Processing completed in {processing_time:.1f}s")
        logger.info(f"Elements processed: {stats.processed_elements}/{stats.total_elements}")
        logger.info(f"Embeddings: {stats.embeddings_generated} new, {stats.embeddings_cached} cached")
        logger.info(f"Estimated cost: ${stats.cost_estimate:.4f}")
        
        return stats
    
    def _optimize_elements(self, elements: List) -> List:
        """Optimize elements by combining, filtering, and deduplicating"""
        optimized = []
        seen_content = set() if self.skip_duplicate_content else None
        
        # Group elements by type for processing
        text_elements = []
        table_elements = []
        figure_elements = []
        
        for element in elements:
            if not element.content or len(element.content.strip()) < self.min_text_length:
                continue
                
            # Skip duplicates
            if seen_content is not None:
                content_hash = hash(element.content.strip().lower())
                if content_hash in seen_content:
                    logger.debug(f"Skipping duplicate content: {element.element_id}")
                    continue
                seen_content.add(content_hash)
            
            # Truncate very long content
            if len(element.content) > self.max_text_length:
                element.content = element.content[:self.max_text_length] + "..."
                logger.debug(f"Truncated long content: {element.element_id}")
            
            # Group by type
            if element.element_type in ["text", "heading"]:
                text_elements.append(element)
            elif element.element_type == "table":
                table_elements.append(element)
            elif element.element_type in ["figure", "diagram"]:
                figure_elements.append(element)
            else:
                text_elements.append(element)  # Default to text
        
        # Combine small adjacent text blocks if enabled
        if self.combine_small_blocks and text_elements:
            text_elements = self._combine_small_text_blocks(text_elements)
        
        # Add all optimized elements
        optimized.extend(text_elements)
        optimized.extend(table_elements)
        optimized.extend(figure_elements)
        
        logger.info(f"Optimized {len(elements)} elements to {len(optimized)} elements")
        return optimized
    
    def _combine_small_text_blocks(self, text_elements: List) -> List:
        """Combine small adjacent text blocks to reduce fragmentation"""
        if not text_elements:
            return text_elements
        
        # Sort by page and position
        text_elements.sort(key=lambda x: (x.page_number, x.bbox[1] if x.bbox else 0))
        
        combined = []
        current_group = [text_elements[0]]
        
        for element in text_elements[1:]:
            prev_element = current_group[-1]
            
            # Combine if:
            # 1. Same page
            # 2. Both are small (< 200 chars)
            # 3. Similar element types
            should_combine = (
                element.page_number == prev_element.page_number and
                len(prev_element.content) < 200 and
                len(element.content) < 200 and
                element.element_type == prev_element.element_type
            )
            
            if should_combine:
                current_group.append(element)
            else:
                # Process current group
                if len(current_group) > 1:
                    combined_element = self._merge_text_elements(current_group)
                    combined.append(combined_element)
                else:
                    combined.append(current_group[0])
                
                current_group = [element]
        
        # Handle last group
        if len(current_group) > 1:
            combined_element = self._merge_text_elements(current_group)
            combined.append(combined_element)
        else:
            combined.append(current_group[0])
        
        if len(combined) < len(text_elements):
            logger.info(f"Combined {len(text_elements)} text blocks into {len(combined)} blocks")
        
        return combined
    
    def _merge_text_elements(self, elements: List) -> 'DocumentElement':
        """Merge multiple text elements into one"""
        # Import here to avoid circular imports
        from app_enhanced import DocumentElement
        
        # Combine content
        combined_content = " ".join(elem.content.strip() for elem in elements)
        
        # Use first element as base
        base_element = elements[0]
        
        # Create new merged element
        merged_element = DocumentElement(
            element_id=f"{base_element.element_id}_merged_{len(elements)}",
            element_type=base_element.element_type,
            page_number=base_element.page_number,
            bbox=base_element.bbox,
            content=combined_content,
            metadata={
                **base_element.metadata,
                'merged_elements': len(elements),
                'original_ids': ', '.join([elem.element_id for elem in elements])  # Convert list to string
            }
        )
        
        return merged_element
    
    def _store_elements_batch(self, elements: List) -> Dict[str, int]:
        """Store elements in appropriate collections with batch operations"""
        stored_counts = {"text": 0, "tables": 0, "figures": 0, "errors": 0}
        
        # Group elements by collection
        collection_groups = {
            "text": [],
            "tables": [],
            "figures": []
        }
        
        for element in elements:
            if not element.text_embedding:
                stored_counts["errors"] += 1
                continue
            
            # Determine collection
            if element.element_type in ["text", "heading"]:
                collection_groups["text"].append(element)
            elif element.element_type == "table":
                collection_groups["tables"].append(element)
            elif element.element_type in ["figure", "diagram"]:
                collection_groups["figures"].append(element)
            else:
                collection_groups["text"].append(element)
        
        # Batch store in each collection
        for group_name, group_elements in collection_groups.items():
            if not group_elements:
                continue
                
            collection = self.collections.get(f"service_manual_{group_name}")
            if not collection:
                logger.error(f"Collection for {group_name} not available")
                stored_counts["errors"] += len(group_elements)
                continue
            
            try:
                # Prepare batch data
                ids = []
                documents = []
                embeddings = []
                metadatas = []
                
                for element in group_elements:
                    # Prepare metadata
                    metadata = element.metadata.copy()
                    metadata.update({
                        "element_type": element.element_type,
                        "page_number": element.page_number,
                        "bbox": str(element.bbox),
                        "element_id": element.element_id,
                        "processing_method": "batch_optimized"
                    })
                    
                    # Add type-specific metadata
                    if element.element_type == "table" and hasattr(element, 'table_data') and element.table_data:
                        metadata["table_html"] = element.table_data.get("html", "")
                        metadata["table_csv"] = element.table_data.get("csv", "")
                    
                    ids.append(element.element_id)
                    documents.append(element.content)
                    embeddings.append(element.text_embedding)
                    metadatas.append(metadata)
                
                # Batch add to collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                stored_counts[group_name] = len(group_elements)
                logger.info(f"Stored {len(group_elements)} elements in {group_name} collection")
                
            except Exception as e:
                logger.error(f"Error storing {group_name} elements: {e}")
                stored_counts["errors"] += len(group_elements)
        
        return stored_counts
    
    def intelligent_search(self, query: str, search_strategy: Dict[str, Any], n_results: int = 5) -> List[Dict]:
        """Enhanced search with fallback to original method"""
        # Use the original intelligent search method from the enhanced processor
        # This maintains compatibility while adding batch processing for uploads
        local_results = []
        
        # Search each collection based on query characteristics
        query_lower = query.lower()
        
        # Search text collection for general queries
        if not any(keyword in query_lower for keyword in ["diagram", "figure", "table", "chart", "spec"]):
            text_results = self._search_collection(self.text_collection, query, n_results//2)
            local_results.extend(text_results)
        
        # Search table collection for specification queries
        if any(keyword in query_lower for keyword in 
               ['spec', 'specification', 'table', 'rating', 'dimension', 'measurement', 'value', 'torque']):
            table_results = self._search_collection(self.table_collection, query, n_results//2)
            local_results.extend(table_results)
        
        # Search figure collection for visual queries  
        if any(keyword in query_lower for keyword in
               ['diagram', 'figure', 'wiring', 'schematic', 'drawing', 'layout', 'connection', 'part']):
            figure_results = self._search_collection(self.figure_collection, query, n_results//2)
            local_results.extend(figure_results)
        
        # If no specific type detected, search all collections
        if not local_results:
            text_results = self._search_collection(self.text_collection, query, n_results//3)
            table_results = self._search_collection(self.table_collection, query, n_results//3)
            figure_results = self._search_collection(self.figure_collection, query, n_results//3)
            local_results.extend(text_results + table_results + figure_results)
        
        # Sort by relevance score and return top results
        local_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return local_results[:n_results]
    
    def _search_collection(self, collection, query: str, n_results: int) -> List[Dict]:
        """Search a specific collection"""
        try:
            if not collection or collection.count() == 0:
                return []
            
            # Get single embedding for query (this is acceptable since it's just one)
            query_embedding = self.embedding_optimizer.get_embeddings_batch([query])[0]
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, collection.count()),
                include=["documents", "distances", "metadatas"]
            )
            
            formatted_results = []
            if results["documents"]:
                for i, (doc, distance, metadata) in enumerate(zip(
                    results["documents"][0],
                    results["distances"][0], 
                    results["metadatas"][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "similarity": 1 - distance,
                        "metadata": metadata,
                        "element_type": metadata.get("element_type", "text")
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search error in collection: {e}")
            return []
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization and cache statistics"""
        cache_stats = self.embedding_optimizer.get_cache_stats()
        
        collection_stats = {}
        for name, collection in self.collections.items():
            if collection:
                collection_stats[name] = collection.count()
            else:
                collection_stats[name] = 0
        
        return {
            'cache_stats': cache_stats,
            'collection_stats': collection_stats,
            'optimization_settings': {
                'min_text_length': self.min_text_length,
                'max_text_length': self.max_text_length,
                'combine_small_blocks': self.combine_small_blocks,
                'skip_duplicate_content': self.skip_duplicate_content
            }
        }
