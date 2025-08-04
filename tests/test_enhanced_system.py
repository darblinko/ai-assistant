#!/usr/bin/env python3
"""
Test script for the enhanced multimodal RAG system with agent integration
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent_pipeline import AgentPipeline
from config.agent_config import load_agent_config, validate_agent_configuration
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

def test_configuration():
    """Test agent configuration loading and validation"""
    print("🧪 Testing Agent Configuration...")
    
    try:
        # Test configuration loading
        config = load_agent_config()
        print("✅ Configuration loaded successfully")
        
        # Test configuration validation
        validation = validate_agent_configuration()
        if validation['valid']:
            print("✅ Configuration validation passed")
            if validation['warnings']:
                print("⚠️  Warnings:")
                for warning in validation['warnings']:
                    print(f"   • {warning}")
        else:
            print("❌ Configuration validation failed:")
            for issue in validation['issues']:
                print(f"   • {issue}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_agent_pipeline():
    """Test the agent pipeline with sample queries"""
    print("\n🤖 Testing Agent Pipeline...")
    
    try:
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if not openai_client.api_key:
            print("❌ OpenAI API key not configured")
            return False
        
        # Initialize agent pipeline
        agent_config = load_agent_config()
        pipeline = AgentPipeline(openai_client, agent_config)
        print("✅ Agent pipeline initialized")
        
        # Test queries - including the problematic automotive query
        test_queries = [
            "how do i replace the alternator on my 2019 Infiniti Q50?",
            "show me the wiring diagram for the headlights",
            "what are the torque specifications for the wheel bolts?",
            "help me troubleshoot engine overheating"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            
            try:
                # Validate query
                validation = pipeline.validate_query(query)
                if not validation['valid']:
                    print(f"❌ Query validation failed: {validation['issues']}")
                    continue
                
                # Process query
                result = await pipeline.process_query(query)
                
                if result['success']:
                    print("✅ Pipeline processing successful")
                    
                    # Display analysis results
                    intent = result.get('intent_analysis', {})
                    equipment = result.get('equipment_analysis', {})
                    strategy = result.get('search_strategy', {})
                    
                    print(f"   🎯 Intent: {intent.get('primary_intent', 'unknown')} "
                          f"(confidence: {intent.get('confidence', 0):.2f})")
                    
                    if equipment:
                        print(f"   🔧 Equipment: {equipment.get('category', 'unknown')} - "
                              f"{equipment.get('brand', '')} {equipment.get('model', '')}")
                    
                    if strategy:
                        print(f"   🔍 Search Strategy: {strategy.get('search_scope', 'unknown')} "
                              f"({len(strategy.get('primary_keywords', []))} keywords)")
                        print(f"   📊 Content Priority: {', '.join(strategy.get('content_type_priority', [])[:2])}")
                    
                    print(f"   ⚡ Processing Time: {result['processing_time']:.2f}s")
                    
                else:
                    print(f"❌ Pipeline processing failed: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"❌ Query processing error: {e}")
        
        # Test pipeline status
        print(f"\n📊 Testing Pipeline Status...")
        status = await pipeline.get_pipeline_status()
        print(f"Pipeline Ready: {status['pipeline_ready']}")
        for agent_name, agent_status in status.get('agents', {}).items():
            ready = agent_status.get('ready', agent_status.get('error') is None)
            print(f"   {agent_name}: {'✅ Ready' if ready else '❌ Failed'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent pipeline test failed: {e}")
        return False

def test_chromadb_collections():
    """Test ChromaDB collection initialization"""
    print("\n📚 Testing ChromaDB Collections...")
    
    try:
        import chromadb
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_db_path = os.path.join(script_dir, 'chroma_db')
        
        client = chromadb.PersistentClient(path=chroma_db_path)
        print("✅ ChromaDB client initialized")
        
        # Test collection creation/access
        collection_names = ["service_manual_text", "service_manual_tables", "service_manual_figures"]
        
        for name in collection_names:
            try:
                collection = client.get_collection(name)
                print(f"✅ Found existing collection: {name} ({collection.count()} items)")
            except:
                try:
                    collection = client.create_collection(
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    print(f"✅ Created new collection: {name}")
                except Exception as e:
                    print(f"❌ Failed to create collection {name}: {e}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_enhanced_processor():
    """Test the enhanced multimodal processor"""
    print("\n🔧 Testing Enhanced Multimodal Processor...")
    
    try:
        from app_enhanced import EnhancedMultimodalProcessor
        from openai import OpenAI
        import chromadb
        
        # Initialize components
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chroma_db_path = os.path.join(script_dir, 'chroma_db')
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Initialize processor
        processor = EnhancedMultimodalProcessor(openai_client, chroma_client)
        print("✅ Enhanced multimodal processor initialized")
        
        # Test collection initialization
        valid_collections = sum(1 for c in processor.collections.values() if c is not None)
        print(f"✅ Collections initialized: {valid_collections}/3")
        
        # Test clear irrelevant data function
        print("🧹 Testing irrelevant data clearing...")
        cleared_counts = processor.clear_irrelevant_data("automotive")
        total_cleared = sum(cleared_counts.values())
        print(f"✅ Cleared {total_cleared} irrelevant documents")
        if total_cleared > 0:
            print(f"   Details: {cleared_counts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced processor test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Enhanced System Test")
    print("=" * 60)
    
    # Test results tracking
    results = {
        'configuration': False,
        'chromadb': False,
        'enhanced_processor': False,
        'agent_pipeline': False
    }
    
    # Run tests
    results['configuration'] = test_configuration()
    results['chromadb'] = test_chromadb_collections()
    results['enhanced_processor'] = test_enhanced_processor()
    results['agent_pipeline'] = await test_agent_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for component, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {component.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python app_enhanced.py")
        print("2. Open: http://localhost:5000")
        print("3. Try query: 'how do i replace the alternator on my 2019 Infiniti Q50?'")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_comprehensive_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with unexpected error: {e}")
        sys.exit(1)
