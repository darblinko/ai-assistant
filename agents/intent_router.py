#!/usr/bin/env python3
"""
Intent Router Agent for Multimodal RAG Assistant
Classifies user queries into actionable intents for optimal processing
"""

import time
import json
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentResponse, AgentError


class IntentRouterAgent(BaseAgent):
    """Classifies user queries into actionable intents"""
    
    INTENT_CATEGORIES = {
        "local_search_only": "Query can be answered with existing documents",
        "manual_search_needed": "Query requires external service manual",
        "general_knowledge": "General question not requiring specific manuals",
        "equipment_specific": "Query about specific equipment requiring classification",
        "troubleshooting": "Diagnostic or repair-related query",
        "maintenance": "Routine maintenance or service query",
        "specifications": "Technical specifications or measurements query",
        "parts_diagram": "Request for visual diagrams or parts breakdown",
        "wiring_electrical": "Electrical wiring or circuit diagrams",
        "installation": "Installation or assembly procedures"
    }
    
    def __init__(self, openai_client, config=None):
        super().__init__(openai_client, config)
        
        # Intent router specific configuration
        self.router_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 300,
            "confidence_threshold": 0.8
        }
        
        # Update effective config
        self.effective_config.update(self.router_config)
        if config:
            self.effective_config.update(config)
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Classify user query into actionable intents
        
        Args:
            input_data: Dict containing 'query' and optional 'context'
            
        Returns:
            AgentResponse with intent classification data
        """
        start_time = time.time()
        
        # Validate input
        missing_keys = self.validate_input(input_data, ["query"])
        if missing_keys:
            return self._create_response(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Missing required keys: {missing_keys}",
                processing_time=time.time() - start_time,
                errors=[f"Missing required input: {', '.join(missing_keys)}"]
            )
        
        try:
            query = input_data["query"].strip()
            context = input_data.get("context", "")
            
            self.logger.info(f"Classifying intent for query: {query[:100]}...")
            
            # Classify the intent
            intent_result = await self.classify_intent(query, context)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Intent classification complete: {intent_result['primary_intent']} "
                           f"(confidence: {intent_result['confidence']:.2f})")
            
            return self._create_response(
                success=True,
                data=intent_result,
                confidence=intent_result["confidence"],
                reasoning=intent_result["reasoning"],
                processing_time=processing_time,
                metadata={
                    "query_length": len(query),
                    "has_context": bool(context),
                    "processing_method": "llm_classification"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return self._create_response(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Error during intent classification: {str(e)}",
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def classify_intent(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Perform deep intent classification using LLM
        
        Args:
            query: User query string
            context: Additional context (optional)
            
        Returns:
            Dict with classification results
        """
        
        # Create intent classification prompt
        intent_descriptions = "\n".join([
            f"- {intent}: {description}" 
            for intent, description in self.INTENT_CATEGORIES.items()
        ])
        
        prompt = f"""You are an expert at classifying service manual queries. Analyze the user's query and classify it into the most appropriate intent categories.

AVAILABLE INTENT CATEGORIES:
{intent_descriptions}

USER QUERY: "{query}"
{f'CONTEXT: {context}' if context else ''}

Analyze this query and provide a JSON response with the following structure:
{{
    "primary_intent": "most_appropriate_category",
    "secondary_intents": ["additional_relevant_categories"],
    "confidence": 0.95,
    "requires_manual": true/false,
    "requires_equipment_classification": true/false,
    "query_complexity": "simple/moderate/complex",
    "reasoning": "Brief explanation of the classification decision",
    "keywords_detected": ["key", "terms", "found"],
    "content_types_needed": ["text", "tables", "diagrams"]
}}

Focus on:
1. The primary action the user wants to perform
2. Whether external manuals are needed
3. If equipment-specific information is required
4. The type of content that would best answer the query

Provide ONLY the JSON response, no additional text."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert query classifier for service manual systems. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self._call_openai(messages)
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")
            
            # Validate and normalize the result
            result = self._validate_classification_result(result, query)
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            # Fallback to rule-based classification
            return self._fallback_classification(query)
    
    def _validate_classification_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and normalize the classification result"""
        
        # Ensure primary_intent is valid
        if result.get("primary_intent") not in self.INTENT_CATEGORIES:
            result["primary_intent"] = "general_knowledge"
        
        # Ensure secondary_intents is a list and contains valid intents
        secondary = result.get("secondary_intents", [])
        if not isinstance(secondary, list):
            secondary = []
        result["secondary_intents"] = [
            intent for intent in secondary 
            if intent in self.INTENT_CATEGORIES and intent != result["primary_intent"]
        ]
        
        # Ensure confidence is in valid range
        confidence = result.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            confidence = 0.7
        result["confidence"] = confidence
        
        # Set default values for missing fields
        result.setdefault("requires_manual", False)
        result.setdefault("requires_equipment_classification", False)
        result.setdefault("query_complexity", "moderate")
        result.setdefault("reasoning", "Classified based on query content")
        result.setdefault("keywords_detected", [])
        result.setdefault("content_types_needed", ["text"])
        
        # Add query metadata
        result["original_query"] = query
        result["query_length"] = len(query)
        
        return result
    
    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """Rule-based fallback classification when LLM fails"""
        
        query_lower = query.lower()
        
        # Equipment-specific patterns
        equipment_patterns = [
            "infiniti", "honda", "toyota", "ford", "chevrolet", "nissan",
            "model", "year", "engine", "transmission", "alternator", "compressor"
        ]
        
        # Specification patterns
        spec_patterns = [
            "torque", "specification", "measurement", "size", "rating",
            "voltage", "amperage", "pressure", "temperature"
        ]
        
        # Diagram patterns
        diagram_patterns = [
            "diagram", "wiring", "schematic", "drawing", "layout",
            "exploded view", "parts breakdown", "circuit"
        ]
        
        # Troubleshooting patterns
        troubleshoot_patterns = [
            "repair", "fix", "troubleshoot", "diagnose", "problem",
            "not working", "broken", "error", "fault"
        ]
        
        # Determine primary intent
        primary_intent = "general_knowledge"
        confidence = 0.6
        secondary_intents = []
        content_types = ["text"]
        requires_equipment = False
        
        if any(pattern in query_lower for pattern in equipment_patterns):
            primary_intent = "equipment_specific"
            requires_equipment = True
            confidence = 0.8
            secondary_intents.append("manual_search_needed")
        
        if any(pattern in query_lower for pattern in spec_patterns):
            if primary_intent == "general_knowledge":
                primary_intent = "specifications"
            else:
                secondary_intents.append("specifications")
            content_types.append("tables")
            confidence = max(confidence, 0.8)
        
        if any(pattern in query_lower for pattern in diagram_patterns):
            if primary_intent == "general_knowledge":
                primary_intent = "parts_diagram"
            else:
                secondary_intents.append("parts_diagram")
            content_types.append("diagrams")
            confidence = max(confidence, 0.8)
        
        if any(pattern in query_lower for pattern in troubleshoot_patterns):
            if primary_intent == "general_knowledge":
                primary_intent = "troubleshooting"
            else:
                secondary_intents.append("troubleshooting")
            confidence = max(confidence, 0.7)
        
        return {
            "primary_intent": primary_intent,
            "secondary_intents": list(set(secondary_intents)),
            "confidence": confidence,
            "requires_manual": primary_intent in ["equipment_specific", "manual_search_needed"],
            "requires_equipment_classification": requires_equipment,
            "query_complexity": "simple" if len(query) < 50 else "moderate",
            "reasoning": "Fallback rule-based classification",
            "keywords_detected": [
                pattern for pattern in equipment_patterns + spec_patterns + diagram_patterns
                if pattern in query_lower
            ],
            "content_types_needed": list(set(content_types)),
            "original_query": query,
            "query_length": len(query)
        }
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of an intent"""
        return self.INTENT_CATEGORIES.get(intent, "Unknown intent")
    
    def should_route_to_equipment_classifier(self, intent_result: Dict[str, Any]) -> bool:
        """Determine if query should be routed to equipment classifier"""
        return (
            intent_result.get("requires_equipment_classification", False) or
            intent_result.get("primary_intent") == "equipment_specific" or
            "equipment_specific" in intent_result.get("secondary_intents", [])
        )
    
    def should_search_external_manuals(self, intent_result: Dict[str, Any]) -> bool:
        """Determine if external manual search is needed"""
        return (
            intent_result.get("requires_manual", False) or
            intent_result.get("primary_intent") == "manual_search_needed" or
            "manual_search_needed" in intent_result.get("secondary_intents", [])
        )
    
    def get_content_type_priorities(self, intent_result: Dict[str, Any]) -> List[str]:
        """Get prioritized list of content types for search"""
        content_types = intent_result.get("content_types_needed", ["text"])
        primary_intent = intent_result.get("primary_intent", "general_knowledge")
        
        # Reorder based on primary intent
        if primary_intent == "specifications":
            content_types = ["tables", "text", "diagrams"]
        elif primary_intent in ["parts_diagram", "wiring_electrical"]:
            content_types = ["diagrams", "text", "tables"]
        elif primary_intent == "troubleshooting":
            content_types = ["text", "diagrams", "tables"]
        
        return list(dict.fromkeys(content_types))  # Remove duplicates while preserving order
