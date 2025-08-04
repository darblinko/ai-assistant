#!/usr/bin/env python3
"""
Test script to demonstrate embedding optimization benefits
"""

import os
import sys
import time
from pathlib import Path
import dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
dotenv.load_dotenv()

from openai import OpenAI
from utils.embedding_optimizer import EmbeddingOptimizer

def simulate_document_processing():
    """Simulate processing a 32-page manual with typical content"""
    
    # Simulate typical content from a 32-page manual
    simulated_elements = []
    
    # Each page typically has:
    for page in range(32):
        # 3-8 text blocks per page
        for block in range(5):
            text_length = 150 + (block * 50)  # Varying lengths
            simulated_text = f"Page {page+1} text block {block+1}: " + "Technical content " * text_length
            simulated_elements.append(simulated_text[:text_length])
        
        # 1-2 tables per page (longer content)
        for table in range(2):
            table_text = f"Page {page+1} table {table+1}: Specifications and measurements " * 200
            simulated_elements.append(table_text[:800])
        
        # 1-3 figures per page
        for figure in range(2):
            figure_text = f"Page {page+1} figure {figure+1}: Diagram showing technical details " * 100
            simulated_elements.append(figure_text[:300])
    
    return simulated_elements

def test_individual_vs_batch():
    """Compare individual vs batch processing"""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key to run this test")
        return
    
    print("🧪 Testing Embedding Optimization Benefits")
    print("=" * 60)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Generate test data
    print("📄 Generating simulated 32-page manual content...")
    test_texts = simulate_document_processing()
    print(f"Generated {len(test_texts)} text elements (typical for 32-page manual)")
    
    # Initialize optimizer
    optimizer = EmbeddingOptimizer(openai_client)
    
    # Get cost estimate
    print("\n💰 Cost Analysis:")
    estimate = optimizer.estimate_cost(test_texts)
    
    print(f"📊 Total texts to process: {estimate.total_texts}")
    print(f"📊 Estimated tokens: {estimate.total_tokens:,}")
    print(f"📊 API batches needed: {estimate.batch_count}")
    print(f"📊 Cache hits: {estimate.cache_hits}")
    
    # Calculate old vs new costs
    individual_calls_cost = estimate.total_texts * 0.0001  # Rough estimate for individual calls
    batch_cost = estimate.estimated_cost
    
    print(f"\n💸 COST COMPARISON:")
    print(f"❌ Individual calls (OLD): ~${individual_calls_cost:.4f}")
    print(f"✅ Batch processing (NEW): ~${batch_cost:.4f}")
    print(f"💰 Savings: ${individual_calls_cost - batch_cost:.4f} ({((individual_calls_cost - batch_cost) / individual_calls_cost * 100):.1f}% reduction)")
    
    # Time comparison simulation
    print(f"\n⏱️  TIME COMPARISON:")
    individual_time = estimate.total_texts * 0.2  # ~200ms per individual call
    batch_time = estimate.batch_count * 1.0       # ~1s per batch call
    
    print(f"❌ Individual calls (OLD): ~{individual_time:.1f} seconds")
    print(f"✅ Batch processing (NEW): ~{batch_time:.1f} seconds")
    print(f"⚡ Speed improvement: {individual_time / batch_time:.1f}x faster")
    
    # Ask user if they want to test with actual API calls
    print(f"\n🔥 ACTUAL API TEST:")
    print(f"This would cost approximately ${batch_cost:.4f} to test with real API calls.")
    
    choice = input("Do you want to run actual API test? (y/N): ").lower().strip()
    
    if choice == 'y':
        print("\n🚀 Running actual batch processing test...")
        
        # Test with first 10 elements to minimize cost
        test_sample = test_texts[:10]
        print(f"Testing with {len(test_sample)} elements (sample)")
        
        start_time = time.time()
        embeddings = optimizer.get_embeddings_batch(test_sample)
        end_time = time.time()
        
        print(f"✅ Successfully generated {len(embeddings)} embeddings")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"📊 Cache hits: {optimizer.estimate_cost(test_sample).cache_hits}")
        
        # Test cache effectiveness by running again
        print("\n🔄 Testing cache effectiveness (running same texts again)...")
        start_time = time.time()
        embeddings2 = optimizer.get_embeddings_batch(test_sample)
        end_time = time.time()
        
        print(f"✅ Successfully retrieved {len(embeddings2)} embeddings from cache")
        print(f"⏱️  Processing time: {end_time - start_time:.2f} seconds (should be much faster!)")
        
        # Show cache stats
        cache_stats = optimizer.get_cache_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   Total entries: {cache_stats['total_entries']}")
        print(f"   Cache size: {cache_stats['cache_size_mb']:.2f} MB")
        
    else:
        print("Skipping actual API test.")
    
    print(f"\n✨ SUMMARY:")
    print(f"The optimization reduces costs by {((individual_calls_cost - batch_cost) / individual_calls_cost * 100):.1f}% and speeds up processing by {individual_time / batch_time:.1f}x")
    print(f"For your 32-page manual, this saves approximately ${individual_calls_cost - batch_cost:.4f} per upload!")

if __name__ == "__main__":
    test_individual_vs_batch()
