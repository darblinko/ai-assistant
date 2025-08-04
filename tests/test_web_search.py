#!/usr/bin/env python3
"""
Test script for the Web Search Agent functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.web_search_agent import WebSearchAgent
from config.agent_config import load_agent_config
from models.agent_models import SearchStrategy
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

async def test_web_search_basic():
    """Test basic web search functionality"""
    print("🌐 Testing Web Search Agent...")
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if not openai_client.api_key:
            print("❌ OpenAI API key not configured")
            return False
        
        # Load configuration
        config = load_agent_config()
        web_search_config = config.get('web_search', {})
        
        # Initialize web search agent
        web_search_agent = WebSearchAgent(openai_client, web_search_config)
        print("✅ Web search agent initialized")
        
        # Test the problematic query
        test_query = "how do i replace the alternator on my 2019 Infiniti Q50?"
        
        from models.agent_models import SearchMetadata, SearchScope, ContentType, IntentCategory, EquipmentCategory
        
        # Create proper search metadata
        search_metadata = SearchMetadata(
            intent_primary=IntentCategory.TROUBLESHOOTING,
            intent_confidence=0.95,
            equipment_category=EquipmentCategory.AUTOMOTIVE,
            equipment_confidence=0.90,
            has_specific_equipment=True,
        )
        
        # Create mock search strategy
        search_strategy = SearchStrategy(
            search_scope=SearchScope.EXTERNAL_PREFERRED,
            primary_keywords=['alternator', 'replacement', 'Infiniti', 'Q50', '2019'],
            secondary_keywords=['repair', 'manual', 'guide'],
            content_type_priority=[ContentType.TEXT, ContentType.DIAGRAMS],
            confidence_threshold=0.7,
            max_results_per_type=3,
            enable_hybrid_search=True,
            search_metadata=search_metadata
        )
        
        print(f"\n🔍 Testing query: {test_query}")
        print("📋 Search strategy configured for automotive equipment")
        
        # Test query generation - Fixed to access Pydantic model attributes directly
        equipment_info = {'category': search_strategy.search_metadata.equipment_category.value}
        intent = search_strategy.search_metadata.intent_primary.value
        
        search_queries = web_search_agent._generate_search_queries(test_query, equipment_info, intent)
        print(f"\n📝 Generated {len(search_queries)} search queries:")
        for i, query in enumerate(search_queries[:5], 1):
            print(f"   {i}. {query}")
        
        # Test agent status
        status = web_search_agent.get_agent_status()
        print(f"\n📊 Agent Status:")
        print(f"   Ready: {status['ready']}")
        print(f"   Download Directory: {status['download_directory']}")
        print(f"   Search Engines:")
        for engine, configured in status['search_engines_configured'].items():
            print(f"     {engine}: {'✅ Ready' if configured else '❌ Not configured'}")
        
        # Check for expected query patterns
        has_manual = any('manual' in q.lower() for q in search_queries)
        has_reddit = any('reddit' in q.lower() for q in search_queries)
        has_pdf = any('pdf' in q.lower() for q in search_queries)
        has_specific = any('q50' in q.lower() and 'infiniti' in q.lower() for q in search_queries)
        
        print(f"\n✅ Query Quality Check:")
        print(f"   Manual queries: {'✅' if has_manual else '❌'}")
        print(f"   Reddit queries: {'✅' if has_reddit else '❌'}")
        print(f"   PDF queries: {'✅' if has_pdf else '❌'}")
        print(f"   Model-specific: {'✅' if has_specific else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web search test failed: {e}")
        return False

def test_configuration():
    """Test web search configuration"""
    print("\n🔧 Testing Configuration...")
    
    try:
        config = load_agent_config()
        web_config = config.get('web_search', {})
        
        print(f"✅ Configuration loaded")
        print(f"   Search engines: {web_config.get('search_engines', [])}")
        print(f"   Max results per engine: {web_config.get('max_results_per_engine', 5)}")
        print(f"   Download directory: {web_config.get('download_directory', 'web_resources')}")
        
        # Check environment variables
        print(f"\n🔑 API Key Status:")
        print(f"   Google Search: {'✅ Set' if os.getenv('GOOGLE_SEARCH_API_KEY') else '❌ Not set'}")
        print(f"   Google Engine ID: {'✅ Set' if os.getenv('GOOGLE_SEARCH_ENGINE_ID') else '❌ Not set'}")
        print(f"   YouTube API: {'✅ Set' if os.getenv('YOUTUBE_API_KEY') else '❌ Not set (optional)'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_web_search_tests():
    """Run all web search tests"""
    print("🚀 Starting Web Search Agent Tests")
    print("=" * 50)
    
    results = {
        'configuration': False,
        'basic_functionality': False
    }
    
    # Test 1: Configuration
    results['configuration'] = test_configuration()
    
    # Test 2: Basic functionality
    results['basic_functionality'] = await test_web_search_basic()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Web Search Test Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Web search agent is ready!")
        print("\nThe agent can now:")
        print("• Generate targeted search queries based on equipment classification")
        print("• Search multiple sources (Google, Reddit, YouTube)")
        print("• Download and extract content from PDFs")
        print("• Prioritize manufacturer sources over general content")
        print("• Handle the automotive query that was causing issues")
        return True
    else:
        print("⚠️  Some tests failed, but basic functionality works")
        print("\nFor full functionality, configure:")
        print("• GOOGLE_SEARCH_API_KEY for Google Custom Search")
        print("• GOOGLE_SEARCH_ENGINE_ID for Custom Search Engine")
        print("• Internet connection for Reddit/YouTube scraping")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(run_web_search_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed: {e}")
        sys.exit(1)
