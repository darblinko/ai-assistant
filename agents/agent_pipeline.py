#!/usr/bin/env python3
"""
Agent Pipeline for Multimodal RAG Assistant
Orchestrates multi-agent processing with intelligent query routing
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .base_agent import BaseAgent, AgentResponse, AgentError
from .intent_router import IntentRouterAgent
from .equipment_classifier import EquipmentClassifierAgent
from .web_search_agent import WebSearchAgent


@dataclass
class PipelineStep:
    """Represents a step in the agent pipeline"""
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    response: Optional[AgentResponse] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Calculate step duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class AgentPipeline:
    """
    Orchestrates multi-agent processing pipeline
    
    Pipeline Flow:
    User Query → Intent Router → Equipment Classifier → Search Strategy → Results
    """
    
    def __init__(self, openai_client, config: Optional[Dict[str, Any]] = None):
        self.openai_client = openai_client
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize agents
        self.intent_router = IntentRouterAgent(
            openai_client, 
            config.get("intent_router", {})
        )
        
        self.equipment_classifier = EquipmentClassifierAgent(
            openai_client,
            config.get("equipment_classifier", {})
        )
        
        self.web_search = WebSearchAgent(
            openai_client,
            config.get("web_search", {})
        )
        
        self.logger.info("Agent pipeline initialized with all agents")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up pipeline-specific logging"""
        logger = logging.getLogger("agents.pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def process_query(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Process a query through the complete agent pipeline
        
        Args:
            query: User query string
            context: Additional context (optional)
            
        Returns:
            Dict containing pipeline results and analysis
        """
        pipeline_start = time.time()
        steps = []
        
        self.logger.info(f"Starting pipeline processing for query: {query[:100]}...")
        
        try:
            # Step 1: Intent Classification
            intent_step = PipelineStep("IntentRouter", time.time())
            steps.append(intent_step)
            
            self.logger.info("Step 1: Intent classification")
            intent_response = await self.intent_router.process({
                "query": query,
                "context": context
            })
            
            intent_step.end_time = time.time()
            intent_step.success = intent_response.success
            intent_step.response = intent_response
            
            if not intent_response.success:
                intent_step.error = f"Intent classification failed: {intent_response.errors}"
                self.logger.error(f"Intent classification failed: {intent_response.errors}")
                return self._create_pipeline_result(query, steps, pipeline_start, success=False)
            
            intent_data = intent_response.data
            self.logger.info(f"Intent classified: {intent_data.get('primary_intent')} "
                           f"(confidence: {intent_data.get('confidence', 0):.2f})")
            
            # Step 2: Equipment Classification (if needed)
            equipment_data = {}
            if self.intent_router.should_route_to_equipment_classifier(intent_data):
                equipment_step = PipelineStep("EquipmentClassifier", time.time())
                steps.append(equipment_step)
                
                self.logger.info("Step 2: Equipment classification")
                equipment_response = await self.equipment_classifier.process({
                    "query": query,
                    "intent_data": intent_data
                })
                
                equipment_step.end_time = time.time()
                equipment_step.success = equipment_response.success
                equipment_step.response = equipment_response
                
                if equipment_response.success:
                    equipment_data = equipment_response.data
                    self.logger.info(f"Equipment classified: {equipment_data.get('category')} - "
                                   f"{equipment_data.get('brand', 'Unknown')} "
                                   f"{equipment_data.get('model', 'Unknown')}")
                else:
                    equipment_step.error = f"Equipment classification failed: {equipment_response.errors}"
                    self.logger.warning(f"Equipment classification failed, proceeding without: {equipment_response.errors}")
            else:
                self.logger.info("Step 2: Equipment classification skipped (not required)")
            
            # Step 3: Generate Search Strategy
            strategy_step = PipelineStep("SearchStrategy", time.time())
            steps.append(strategy_step)
            
            self.logger.info("Step 3: Generating search strategy")
            search_strategy = self._generate_search_strategy(intent_data, equipment_data, query)
            
            strategy_step.end_time = time.time()
            strategy_step.success = True
            
            self.logger.info(f"Search strategy generated: {search_strategy['search_scope']} "
                           f"with {len(search_strategy['primary_keywords'])} primary keywords")
            
            # Create final pipeline result
            pipeline_result = self._create_pipeline_result(
                query, steps, pipeline_start, success=True,
                intent_data=intent_data,
                equipment_data=equipment_data,
                search_strategy=search_strategy
            )
            
            total_time = time.time() - pipeline_start
            self.logger.info(f"Pipeline processing complete in {total_time:.2f}s")
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return self._create_pipeline_result(
                query, steps, pipeline_start, success=False,
                error=str(e)
            )
    
    def _generate_search_strategy(
        self, 
        intent_data: Dict[str, Any], 
        equipment_data: Dict[str, Any], 
        query: str
    ) -> Dict[str, Any]:
        """Generate comprehensive search strategy based on agent analysis"""
        
        # Base strategy from intent
        content_priorities = self.intent_router.get_content_type_priorities(intent_data)
        
        # Enhanced strategy from equipment classification
        if equipment_data:
            equipment_strategy = self.equipment_classifier.get_search_strategy(equipment_data)
            primary_keywords = equipment_strategy.get("primary_keywords", [])
            secondary_keywords = equipment_strategy.get("secondary_keywords", [])
            search_scope = equipment_strategy.get("search_scope", "local_first")
            source_priority = equipment_strategy.get("source_priority", [])
        else:
            # Fallback keywords from intent and query
            primary_keywords = intent_data.get("keywords_detected", [])[:5]
            secondary_keywords = []
            search_scope = "local_first"
            source_priority = ["general_manuals"]
            
            # Add query words as secondary keywords
            query_words = [word for word in query.split() if len(word) > 3]
            secondary_keywords.extend(query_words[:5])
        
        # Determine if external search is needed
        needs_external_search = (
            self.intent_router.should_search_external_manuals(intent_data) or
            (equipment_data and equipment_data.get("confidence", 0) > 0.8)
        )
        
        strategy = {
            "search_scope": search_scope,
            "content_type_priority": content_priorities,
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "source_priority": source_priority,
            "needs_external_search": needs_external_search,
            "confidence_threshold": 0.7,
            "max_results_per_type": 5,
            "enable_hybrid_search": True,
            "search_metadata": {
                "intent_primary": intent_data.get("primary_intent"),
                "intent_confidence": intent_data.get("confidence", 0),
                "equipment_category": equipment_data.get("category", "unknown"),
                "equipment_confidence": equipment_data.get("confidence", 0),
                "has_specific_equipment": bool(equipment_data.get("brand") or equipment_data.get("model")),
                "query_complexity": intent_data.get("query_complexity", "moderate")
            }
        }
        
        # Adjust strategy based on query complexity
        if intent_data.get("query_complexity") == "complex":
            strategy["max_results_per_type"] = 8
            strategy["confidence_threshold"] = 0.6
        elif intent_data.get("query_complexity") == "simple":
            strategy["max_results_per_type"] = 3
            strategy["confidence_threshold"] = 0.8
        
        return strategy
    
    def _create_pipeline_result(
        self,
        query: str,
        steps: List[PipelineStep],
        start_time: float,
        success: bool,
        intent_data: Optional[Dict[str, Any]] = None,
        equipment_data: Optional[Dict[str, Any]] = None,
        search_strategy: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standardized pipeline result"""
        
        total_time = time.time() - start_time
        
        # Calculate step summaries
        step_summaries = []
        for step in steps:
            summary = {
                "agent": step.agent_name,
                "duration": step.duration,
                "success": step.success,
                "confidence": step.response.confidence if step.response else 0.0
            }
            if step.error:
                summary["error"] = step.error
            step_summaries.append(summary)
        
        result = {
            "success": success,
            "query": query,
            "processing_time": total_time,
            "steps": step_summaries,
            "agent_count": len(steps),
            "pipeline_version": "1.0"
        }
        
        if success and intent_data:
            result["intent_analysis"] = intent_data
            
        if success and equipment_data:
            result["equipment_analysis"] = equipment_data
            
        if success and search_strategy:
            result["search_strategy"] = search_strategy
            
        if error:
            result["error"] = error
            
        # Add processing metadata
        result["metadata"] = {
            "intent_router_used": any(step.agent_name == "IntentRouter" for step in steps),
            "equipment_classifier_used": any(step.agent_name == "EquipmentClassifier" for step in steps),
            "total_agents_used": len([step for step in steps if step.success]),
            "fastest_step": min(steps, key=lambda s: s.duration).agent_name if steps else None,
            "slowest_step": max(steps, key=lambda s: s.duration).agent_name if steps else None,
            "overall_confidence": self._calculate_overall_confidence(intent_data, equipment_data)
        }
        
        return result
    
    def _calculate_overall_confidence(
        self, 
        intent_data: Optional[Dict[str, Any]], 
        equipment_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall pipeline confidence score"""
        
        confidences = []
        
        if intent_data and intent_data.get("confidence"):
            confidences.append(intent_data["confidence"])
            
        if equipment_data and equipment_data.get("confidence"):
            confidences.append(equipment_data["confidence"])
        
        if not confidences:
            return 0.5  # Default moderate confidence
        
        # Weighted average (intent router is more critical)
        if len(confidences) == 2:
            return (confidences[0] * 0.6) + (confidences[1] * 0.4)
        else:
            return confidences[0]
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and agent health"""
        
        status = {
            "pipeline_ready": True,
            "agents": {},
            "last_updated": time.time()
        }
        
        # Check each agent status
        try:
            status["agents"]["intent_router"] = self.intent_router.get_status()
        except Exception as e:
            status["agents"]["intent_router"] = {"error": str(e), "ready": False}
            status["pipeline_ready"] = False
        
        try:
            status["agents"]["equipment_classifier"] = self.equipment_classifier.get_status()
        except Exception as e:
            status["agents"]["equipment_classifier"] = {"error": str(e), "ready": False}
            status["pipeline_ready"] = False
        
        return status
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate query before processing"""
        
        validation = {
            "valid": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check query length
        if len(query.strip()) < 5:
            validation["valid"] = False
            validation["issues"].append("Query too short (minimum 5 characters)")
        
        if len(query) > 1000:
            validation["valid"] = False
            validation["issues"].append("Query too long (maximum 1000 characters)")
        
        # Check for meaningful content
        if query.strip().isdigit():
            validation["valid"] = False
            validation["issues"].append("Query contains only numbers")
        
        # Recommendations for better results
        query_lower = query.lower()
        
        if not any(word in query_lower for word in ["what", "how", "show", "find", "replace", "repair", "install"]):
            validation["recommendations"].append("Consider starting with 'what', 'how', 'show', etc. for better results")
        
        if not any(word in query_lower for word in ["manual", "service", "repair", "maintenance", "part", "component"]):
            validation["recommendations"].append("Include terms like 'manual', 'service', or 'repair' for more targeted results")
        
        return validation
    
    def get_processing_summary(self, pipeline_result: Dict[str, Any]) -> str:
        """Generate human-readable processing summary"""
        
        if not pipeline_result.get("success"):
            return f"❌ Processing failed: {pipeline_result.get('error', 'Unknown error')}"
        
        intent = pipeline_result.get("intent_analysis", {})
        equipment = pipeline_result.get("equipment_analysis", {})
        strategy = pipeline_result.get("search_strategy", {})
        
        summary_parts = []
        
        # Intent summary
        primary_intent = intent.get("primary_intent", "unknown")
        intent_confidence = intent.get("confidence", 0)
        summary_parts.append(f"🎯 Intent: {primary_intent} ({intent_confidence:.0%} confidence)")
        
        # Equipment summary (if available)
        if equipment:
            category = equipment.get("category", "unknown")
            brand = equipment.get("brand", "")
            model = equipment.get("model", "")
            equipment_info = f"{category}"
            if brand:
                equipment_info += f" - {brand}"
            if model:
                equipment_info += f" {model}"
            summary_parts.append(f"🔧 Equipment: {equipment_info}")
        
        # Search strategy summary
        if strategy:
            search_scope = strategy.get("search_scope", "local_first")
            content_types = strategy.get("content_type_priority", [])
            summary_parts.append(f"🔍 Search: {search_scope} ({', '.join(content_types[:2])} priority)")
        
        # Performance summary
        processing_time = pipeline_result.get("processing_time", 0)
        agent_count = pipeline_result.get("agent_count", 0)
        summary_parts.append(f"⚡ Processed in {processing_time:.2f}s using {agent_count} agents")
        
        return " | ".join(summary_parts)
