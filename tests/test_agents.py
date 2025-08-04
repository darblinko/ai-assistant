#!/usr/bin/env python3
"""
Agent System Test Suite
Comprehensive testing for the intelligent agent framework
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any, List
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.base_agent import BaseAgent, AgentResponse
    from agents.intent_router import IntentRouterAgent
    from agents.equipment_classifier import EquipmentClassifierAgent
    from agents.agent_pipeline import AgentPipeline
    from config.agent_config import load_agent_config, validate_agent_configuration
    from models.agent_models import *
    from openai import OpenAI
    import dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Load environment variables
dotenv.load_dotenv()

class AgentTester:
    """Comprehensive test suite for agent system"""
    
    def __init__(self):
        self.openai_client = None
        self.config = None
        self.test_results = []
        
    def setup(self):
        """Set up test environment"""
        print("🔧 Setting up test environment...")
        
        # Load configuration
        try:
            self.config = load_agent_config()
            print("✅ Configuration loaded")
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return False
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            return False
        
        try:
            self.openai_client = OpenAI(api_key=api_key)
            print("✅ OpenAI client initialized")
        except Exception as e:
            print(f"❌ OpenAI client initialization failed: {e}")
            return False
        
        return True
    
    def record_test_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Record test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time()
        }
        self.test_results.append(result)
        
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
        if not success and "error" in details:
            print(f"   Error: {details['error']}")
    
    async def test_configuration_validation(self):
        """Test configuration validation"""
        print("\n📋 Testing Configuration Validation...")
        
        try:
            validation = validate_agent_configuration()
            
            self.record_test_result(
                "Configuration Validation",
                validation["valid"],
                {
                    "issues": validation.get("issues", []),
                    "warnings": validation.get("warnings", []),
                    "config_loaded": bool(validation.get("config_summary"))
                }
            )
            
            if validation["warnings"]:
                print(f"   ⚠️  {len(validation['warnings'])} warnings found")
                
        except Exception as e:
            self.record_test_result(
                "Configuration Validation",
                False,
                {"error": str(e)}
            )
    
    async def test_intent_router_agent(self):
        """Test Intent Router Agent"""
        print("\n🎯 Testing Intent Router Agent...")
        
        try:
            # Initialize agent
            agent = IntentRouterAgent(
                self.openai_client, 
                self.config["intent_router"]
            )
            
            # Test queries
            test_queries = [
                "How do I replace the alternator on a 2018 Honda Civic?",
                "Show me the wiring diagram for the compressor",
                "What are the torque specifications for the transmission bolts?",
                "I need to troubleshoot my refrigerator",
                "General maintenance schedule question"
            ]
            
            successful_tests = 0
            total_tests = len(test_queries)
            
            for i, query in enumerate(test_queries, 1):
                try:
                    print(f"   Testing query {i}/{total_tests}: {query[:50]}...")
                    
                    response = await agent.process({
                        "query": query,
                        "context": ""
                    })
                    
                    if response.success:
                        intent_data = response.data
                        print(f"     Intent: {intent_data.get('primary_intent')} "
                              f"(confidence: {intent_data.get('confidence', 0):.2f})")
                        successful_tests += 1
                    else:
                        print(f"     Failed: {response.errors}")
                        
                except Exception as e:
                    print(f"     Error: {e}")
            
            self.record_test_result(
                "Intent Router Agent",
                successful_tests == total_tests,
                {
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "success_rate": successful_tests / total_tests
                }
            )
            
        except Exception as e:
            self.record_test_result(
                "Intent Router Agent",
                False,
                {"error": str(e)}
            )
    
    async def test_equipment_classifier_agent(self):
        """Test Equipment Classifier Agent"""
        print("\n🔧 Testing Equipment Classifier Agent...")
        
        try:
            # Initialize agent
            agent = EquipmentClassifierAgent(
                self.openai_client,
                self.config["equipment_classifier"]
            )
            
            # Test queries with equipment-specific content
            test_queries = [
                "2019 Infiniti Q50 oil filter replacement",
                "Samsung refrigerator compressor repair",
                "Caterpillar generator maintenance",
                "HP printer power supply replacement",
                "Carrier HVAC unit thermostat installation"
            ]
            
            successful_tests = 0
            total_tests = len(test_queries)
            
            for i, query in enumerate(test_queries, 1):
                try:
                    print(f"   Testing query {i}/{total_tests}: {query[:50]}...")
                    
                    response = await agent.process({
                        "query": query,
                        "intent_data": {"primary_intent": "equipment_specific"}
                    })
                    
                    if response.success:
                        equipment_data = response.data
                        print(f"     Equipment: {equipment_data.get('category')} - "
                              f"{equipment_data.get('brand', 'Unknown')} "
                              f"{equipment_data.get('model', 'Unknown')} "
                              f"(confidence: {equipment_data.get('confidence', 0):.2f})")
                        successful_tests += 1
                    else:
                        print(f"     Failed: {response.errors}")
                        
                except Exception as e:
                    print(f"     Error: {e}")
            
            self.record_test_result(
                "Equipment Classifier Agent",
                successful_tests == total_tests,
                {
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "success_rate": successful_tests / total_tests
                }
            )
            
        except Exception as e:
            self.record_test_result(
                "Equipment Classifier Agent",
                False,
                {"error": str(e)}
            )
    
    async def test_agent_pipeline(self):
        """Test complete Agent Pipeline"""
        print("\n🔄 Testing Agent Pipeline...")
        
        try:
            # Initialize pipeline
            pipeline = AgentPipeline(self.openai_client, self.config)
            
            # Test comprehensive queries
            test_queries = [
                "How do I replace the alternator on a 2018 Honda Civic?",
                "Show me the torque specifications for transmission bolts",
                "Troubleshoot Samsung refrigerator not cooling",
                "Install thermostat on Carrier HVAC unit",
                "General car maintenance tips"
            ]
            
            successful_tests = 0
            total_tests = len(test_queries)
            
            for i, query in enumerate(test_queries, 1):
                try:
                    print(f"   Testing query {i}/{total_tests}: {query[:50]}...")
                    
                    result = await pipeline.process_query(query)
                    
                    if result["success"]:
                        processing_time = result.get("processing_time", 0)
                        agent_count = result.get("agent_count", 0)
                        
                        print(f"     Processed in {processing_time:.2f}s using {agent_count} agents")
                        
                        # Show pipeline summary
                        summary = pipeline.get_processing_summary(result)
                        print(f"     {summary}")
                        
                        successful_tests += 1
                    else:
                        print(f"     Failed: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"     Error: {e}")
            
            self.record_test_result(
                "Agent Pipeline",
                successful_tests == total_tests,
                {
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "success_rate": successful_tests / total_tests
                }
            )
            
        except Exception as e:
            self.record_test_result(
                "Agent Pipeline",
                False,
                {"error": str(e)}
            )
    
    async def test_pydantic_models(self):
        """Test Pydantic model validation"""
        print("\n📋 Testing Pydantic Models...")
        
        try:
            successful_tests = 0
            total_tests = 0
            
            # Test IntentAnalysis model
            total_tests += 1
            try:
                intent_analysis = IntentAnalysis(
                    primary_intent=IntentCategory.EQUIPMENT_SPECIFIC,
                    secondary_intents=[IntentCategory.TROUBLESHOOTING],
                    confidence=0.85,
                    requires_manual=True,
                    requires_equipment_classification=True,
                    query_complexity=QueryComplexity.MODERATE,
                    reasoning="Equipment-specific query with troubleshooting elements",
                    keywords_detected=["honda", "civic", "alternator"],
                    content_types_needed=[ContentType.DIAGRAMS, ContentType.TEXT],
                    original_query="How to replace Honda Civic alternator",
                    query_length=34
                )
                print("   ✅ IntentAnalysis model validation passed")
                successful_tests += 1
            except Exception as e:
                print(f"   ❌ IntentAnalysis model validation failed: {e}")
            
            # Test EquipmentAnalysis model
            total_tests += 1
            try:
                equipment_analysis = EquipmentAnalysis(
                    category=EquipmentCategory.AUTOMOTIVE,
                    subcategory="passenger_vehicle",
                    brand="Honda",
                    model="Civic",
                    year="2018",
                    component="alternator",
                    task="replacement",
                    confidence=0.92,
                    extracted_entities=ExtractedEntities(
                        brands=["Honda"],
                        models=["Civic"],
                        years=["2018"],
                        components=["alternator"]
                    ),
                    search_keywords=["2018", "Honda", "Civic", "alternator", "replacement"],
                    priority_sources=["manufacturer_sites", "automotive_manuals"],
                    reasoning="Clear automotive equipment specification",
                    original_query="2018 Honda Civic alternator replacement",
                    category_confidence=0.95
                )
                print("   ✅ EquipmentAnalysis model validation passed")
                successful_tests += 1
            except Exception as e:
                print(f"   ❌ EquipmentAnalysis model validation failed: {e}")
            
            # Test SearchStrategy model
            total_tests += 1
            try:
                search_strategy = SearchStrategy(
                    search_scope=SearchScope.EXTERNAL_PREFERRED,
                    content_type_priority=[ContentType.DIAGRAMS, ContentType.TEXT],
                    primary_keywords=["Honda", "Civic", "alternator"],
                    secondary_keywords=["replacement", "2018"],
                    source_priority=["manufacturer_sites"],
                    needs_external_search=True,
                    confidence_threshold=0.7,
                    max_results_per_type=5,
                    enable_hybrid_search=True,
                    search_metadata=SearchMetadata(
                        intent_primary=IntentCategory.EQUIPMENT_SPECIFIC,
                        intent_confidence=0.85,
                        equipment_category=EquipmentCategory.AUTOMOTIVE,
                        equipment_confidence=0.92,
                        has_specific_equipment=True,
                        query_complexity=QueryComplexity.MODERATE
                    )
                )
                print("   ✅ SearchStrategy model validation passed")
                successful_tests += 1
            except Exception as e:
                print(f"   ❌ SearchStrategy model validation failed: {e}")
            
            self.record_test_result(
                "Pydantic Models",
                successful_tests == total_tests,
                {
                    "successful_tests": successful_tests,
                    "total_tests": total_tests,
                    "success_rate": successful_tests / total_tests
                }
            )
            
        except Exception as e:
            self.record_test_result(
                "Pydantic Models",
                False,
                {"error": str(e)}
            )
    
    async def test_error_handling(self):
        """Test error handling and edge cases"""
        print("\n⚠️  Testing Error Handling...")
        
        try:
            # Initialize pipeline
            pipeline = AgentPipeline(self.openai_client, self.config)
            
            error_tests = [
                ("Empty query", ""),
                ("Very short query", "hi"),
                ("Only numbers", "12345"),
                ("Special characters", "!@#$%^&*()"),
                ("Very long query", "a" * 1500)
            ]
            
            successful_error_handling = 0
            total_error_tests = len(error_tests)
            
            for test_name, query in error_tests:
                try:
                    print(f"   Testing {test_name}...")
                    
                    # Validate query first
                    validation = pipeline.validate_query(query)
                    
                    if validation["valid"]:
                        # Process if valid
                        result = await pipeline.process_query(query)
                        if result["success"]:
                            print(f"     Handled gracefully")
                            successful_error_handling += 1
                        else:
                            print(f"     Failed appropriately: {result.get('error', 'Unknown')}")
                            successful_error_handling += 1
                    else:
                        print(f"     Validation caught issues: {validation['issues']}")
                        successful_error_handling += 1
                        
                except Exception as e:
                    print(f"     Error: {e}")
            
            self.record_test_result(
                "Error Handling",
                successful_error_handling == total_error_tests,
                {
                    "successful_tests": successful_error_handling,
                    "total_tests": total_error_tests,
                    "success_rate": successful_error_handling / total_error_tests
                }
            )
            
        except Exception as e:
            self.record_test_result(
                "Error Handling",
                False,
                {"error": str(e)}
            )
    
    async def run_performance_tests(self):
        """Run performance benchmarks"""
        print("\n⚡ Running Performance Tests...")
        
        try:
            pipeline = AgentPipeline(self.openai_client, self.config)
            
            test_query = "How do I replace the alternator on a 2018 Honda Civic?"
            num_runs = 3
            
            times = []
            successful_runs = 0
            
            for i in range(num_runs):
                try:
                    start_time = time.time()
                    result = await pipeline.process_query(test_query)
                    end_time = time.time()
                    
                    if result["success"]:
                        processing_time = end_time - start_time
                        times.append(processing_time)
                        successful_runs += 1
                        print(f"   Run {i+1}: {processing_time:.2f}s")
                    else:
                        print(f"   Run {i+1}: Failed")
                        
                except Exception as e:
                    print(f"   Run {i+1}: Error - {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                print(f"   Average: {avg_time:.2f}s")
                print(f"   Range: {min_time:.2f}s - {max_time:.2f}s")
                
                # Performance thresholds
                performance_good = avg_time < 5.0
                consistency_good = (max_time - min_time) < 3.0
                
                self.record_test_result(
                    "Performance Tests",
                    performance_good and consistency_good,
                    {
                        "successful_runs": successful_runs,
                        "total_runs": num_runs,
                        "average_time": avg_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "performance_threshold_met": performance_good,
                        "consistency_threshold_met": consistency_good
                    }
                )
            else:
                self.record_test_result(
                    "Performance Tests",
                    False,
                    {"error": "No successful runs completed"}
                )
                
        except Exception as e:
            self.record_test_result(
                "Performance Tests",
                False,
                {"error": str(e)}
            )
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print("🧪 AGENT SYSTEM TEST SUMMARY")
        print("="*70)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        
        print(f"\nOverall Results: {successful_tests}/{total_tests} tests passed")
        print(f"Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} - {result['test_name']}")
            
            if "success_rate" in result["details"]:
                rate = result["details"]["success_rate"] * 100
                print(f"    Sub-test success rate: {rate:.1f}%")
        
        print("\n" + "="*70)
        
        if successful_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Agent system is ready for deployment.")
        else:
            print("⚠️  Some tests failed. Review the results above.")
            print("   Check configuration, API keys, and network connectivity.")
        
        print("="*70)
    
    async def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Comprehensive Agent System Tests")
        print("="*70)
        
        if not self.setup():
            print("❌ Test setup failed. Aborting tests.")
            return False
        
        # Run all test categories
        await self.test_configuration_validation()
        await self.test_pydantic_models()
        await self.test_intent_router_agent()
        await self.test_equipment_classifier_agent()
        await self.test_agent_pipeline()
        await self.test_error_handling()
        await self.run_performance_tests()
        
        # Print summary
        self.print_test_summary()
        
        return all(result["success"] for result in self.test_results)


async def main():
    """Main test execution"""
    tester = AgentTester()
    
    try:
        success = await tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if we have required dependencies
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        sys.exit(1)
    
    # Run tests
    asyncio.run(main())
