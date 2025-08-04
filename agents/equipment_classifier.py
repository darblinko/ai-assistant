#!/usr/bin/env python3
"""
Equipment Classifier Agent for Multimodal RAG Assistant
Advanced equipment type and component classification system
"""

import time
import json
import re
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentResponse, AgentError


class EquipmentClassifierAgent(BaseAgent):
    """Advanced equipment type and component classification"""
    
    EQUIPMENT_TAXONOMY = {
        "automotive": {
            "subcategories": ["passenger_vehicle", "commercial_vehicle", "motorcycle", "marine"],
            "common_brands": ["toyota", "honda", "ford", "chevrolet", "nissan", "infiniti", 
                            "bmw", "mercedes", "audi", "volkswagen", "hyundai", "kia", "mazda", 
                            "subaru", "mitsubishi", "lexus", "acura", "cadillac", "buick", "gmc"],
            "components": ["engine", "transmission", "brakes", "electrical", "hvac", "filters",
                         "alternator", "starter", "compressor", "radiator", "battery", "fuel_pump",
                         "oil_pump", "water_pump", "timing_belt", "spark_plugs", "injectors"],
            "tasks": ["replacement", "repair", "diagnosis", "maintenance", "upgrade", "installation"],
            "model_patterns": [
                r"(\d{4})\s+(\w+)\s+(\w+)",  # Year Brand Model
                r"(\w+)\s+(\w+)\s+(\d{4})",  # Brand Model Year
                r"(\d{2,4})\s*(\w+)",       # Year Brand
            ],
            "priority_sources": ["manufacturer_sites", "automotive_manuals", "repair_databases"]
        },
        "appliances": {
            "subcategories": ["kitchen", "laundry", "cooling", "small_appliance", "water_treatment"],
            "common_brands": ["whirlpool", "ge", "samsung", "lg", "maytag", "true", "kitchenaid",
                            "bosch", "frigidaire", "electrolux", "kenmore", "amana", "viking"],
            "components": ["compressor", "motor", "control_board", "door_seal", "filter", 
                         "heating_element", "thermostat", "pump", "valve", "sensor", "timer"],
            "tasks": ["replacement", "repair", "cleaning", "calibration", "troubleshooting", "maintenance"],
            "model_patterns": [
                r"model[:\s]*([a-zA-Z0-9\-]+)",
                r"(\w{2,4}\d{2,6})",  # Brand codes + numbers
                r"part[:\s]*([a-zA-Z0-9\-]+)",
            ],
            "priority_sources": ["appliance_manuals", "manufacturer_support", "parts_catalogs"]
        },
        "industrial": {
            "subcategories": ["pumps", "compressors", "generators", "motors", "controls", "hydraulics"],
            "common_brands": ["caterpillar", "cummins", "ingersoll_rand", "atlas_copco", "siemens",
                            "abb", "schneider", "allen_bradley", "honeywell", "emerson"],
            "components": ["impeller", "bearing", "seal", "controller", "sensor", "actuator",
                         "valve", "coupling", "rotor", "stator", "housing", "gasket"],
            "tasks": ["overhaul", "replacement", "calibration", "preventive_maintenance", 
                     "commissioning", "troubleshooting"],
            "model_patterns": [
                r"([A-Z]{2,4}[-\s]*\d+[A-Z]*)",  # Industrial model codes
                r"series[:\s]*([a-zA-Z0-9\-]+)",
                r"type[:\s]*([a-zA-Z0-9\-]+)",
            ],
            "priority_sources": ["industrial_manuals", "technical_documentation", "oem_support"]
        },
        "electronics": {
            "subcategories": ["computing", "printing", "audio_video", "networking", "telecommunications"],
            "common_brands": ["hp", "canon", "epson", "sony", "panasonic", "cisco", "dell",
                            "lenovo", "apple", "microsoft", "samsung", "lg", "sharp"],
            "components": ["power_supply", "motherboard", "display", "connectivity", "processor",
                         "memory", "storage", "cooling_fan", "antenna", "circuit_board"],
            "tasks": ["replacement", "upgrade", "configuration", "troubleshooting", "firmware_update"],
            "model_patterns": [
                r"([A-Z]{2,6}\d{2,6}[A-Z]*)",  # Electronics model codes
                r"model[:\s]*([a-zA-Z0-9\-]+)",
                r"part[:\s#]*([a-zA-Z0-9\-]+)",
            ],
            "priority_sources": ["electronics_manuals", "manufacturer_support", "technical_specs"]
        },
        "hvac": {
            "subcategories": ["heating", "cooling", "ventilation", "controls", "ductwork"],
            "common_brands": ["carrier", "trane", "lennox", "rheem", "goodman", "york", "daikin",
                            "mitsubishi", "fujitsu", "american_standard", "bryant", "payne"],
            "components": ["compressor", "evaporator", "condenser", "thermostat", "ductwork",
                         "filter", "blower", "heat_exchanger", "expansion_valve", "refrigerant_lines"],
            "tasks": ["replacement", "repair", "cleaning", "seasonal_maintenance", "installation"],
            "model_patterns": [
                r"([A-Z]{2,4}\d{2,6}[A-Z]*)",  # HVAC model codes
                r"model[:\s]*([a-zA-Z0-9\-]+)",
                r"unit[:\s]*([a-zA-Z0-9\-]+)",
            ],
            "priority_sources": ["hvac_manuals", "contractor_resources", "manufacturer_specs"]
        }
    }
    
    def __init__(self, openai_client, config=None):
        super().__init__(openai_client, config)
        
        # Equipment classifier specific configuration
        self.classifier_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 400,
            "enable_entity_extraction": True,
            "confidence_threshold": 0.7
        }
        
        # Update effective config
        self.effective_config.update(self.classifier_config)
        if config:
            self.effective_config.update(config)
    
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Classify equipment and extract detailed information
        
        Args:
            input_data: Dict containing 'query' and optional 'intent_data'
            
        Returns:
            AgentResponse with equipment classification data
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
            intent_data = input_data.get("intent_data", {})
            
            self.logger.info(f"Classifying equipment for query: {query[:100]}...")
            
            # Classify the equipment
            equipment_result = await self.classify_equipment(query, intent_data)
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Equipment classification complete: {equipment_result['category']} - "
                           f"{equipment_result.get('brand', 'Unknown')} {equipment_result.get('model', 'Unknown')} "
                           f"(confidence: {equipment_result['confidence']:.2f})")
            
            return self._create_response(
                success=True,
                data=equipment_result,
                confidence=equipment_result["confidence"],
                reasoning=equipment_result["reasoning"],
                processing_time=processing_time,
                metadata={
                    "query_length": len(query),
                    "has_intent_data": bool(intent_data),
                    "processing_method": "llm_classification_with_taxonomy",
                    "entities_extracted": len(equipment_result.get("extracted_entities", {}))
                }
            )
            
        except Exception as e:
            self.logger.error(f"Equipment classification failed: {e}")
            return self._create_response(
                success=False,
                data={},
                confidence=0.0,
                reasoning=f"Error during equipment classification: {str(e)}",
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def classify_equipment(self, query: str, intent_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform detailed equipment classification using LLM + taxonomy
        
        Args:
            query: User query string
            intent_data: Intent classification data (optional)
            
        Returns:
            Dict with detailed equipment classification
        """
        
        # First, try entity extraction from query
        extracted_entities = self._extract_entities_from_query(query)
        
        # Create equipment classification prompt
        taxonomy_summary = self._create_taxonomy_summary()
        
        intent_context = ""
        if intent_data:
            intent_context = f"""
INTENT CONTEXT:
- Primary Intent: {intent_data.get('primary_intent', 'unknown')}
- Requires Equipment Classification: {intent_data.get('requires_equipment_classification', False)}
- Content Types Needed: {', '.join(intent_data.get('content_types_needed', []))}
"""
        
        prompt = f"""You are an expert equipment classifier for service manuals. Analyze the query to identify specific equipment, components, and tasks.

EQUIPMENT TAXONOMY:
{taxonomy_summary}

USER QUERY: "{query}"
{intent_context}

PRE-EXTRACTED ENTITIES:
{json.dumps(extracted_entities, indent=2)}

Analyze this query and provide a JSON response with the following structure:
{{
    "category": "automotive/appliances/industrial/electronics/hvac",
    "subcategory": "specific_subcategory",
    "brand": "brand_name_if_detected",
    "model": "model_number_if_detected", 
    "year": "year_if_detected",
    "component": "specific_component",
    "task": "specific_task_or_operation",
    "confidence": 0.95,
    "extracted_entities": {{
        "brands": ["detected", "brands"],
        "models": ["detected", "models"],
        "years": ["detected", "years"],
        "components": ["detected", "components"],
        "part_numbers": ["detected", "part_numbers"]
    }},
    "search_keywords": ["prioritized", "search", "terms"],
    "priority_sources": ["recommended", "source", "types"],
    "reasoning": "Detailed explanation of classification decision",
    "alternative_interpretations": ["other", "possible", "classifications"]
}}

Focus on:
1. Identifying the primary equipment category
2. Extracting specific brand, model, year information
3. Determining the component or part involved
4. Understanding the task/operation being performed
5. Generating effective search keywords
6. Recommending appropriate source types

Provide ONLY the JSON response, no additional text."""

        try:
            messages = [
                {"role": "system", "content": "You are an expert equipment classifier. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = await self._call_openai(messages)
            
            # Parse JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")
            
            # Validate and enhance the result
            result = self._validate_and_enhance_classification(result, query, extracted_entities)
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            # Fallback to rule-based classification
            return self._fallback_equipment_classification(query, extracted_entities)
    
    def _extract_entities_from_query(self, query: str) -> Dict[str, List[str]]:
        """Extract equipment entities using regex patterns"""
        
        entities = {
            "years": [],
            "brands": [],
            "models": [],
            "components": [],
            "part_numbers": []
        }
        
        # Extract years (1950-2030)
        year_pattern = r'\b(19[5-9]\d|20[0-3]\d)\b'
        entities["years"] = re.findall(year_pattern, query)
        
        # Extract common brands across all categories
        all_brands = []
        for category_data in self.EQUIPMENT_TAXONOMY.values():
            all_brands.extend(category_data.get("common_brands", []))
        
        query_lower = query.lower()
        for brand in all_brands:
            if brand.lower() in query_lower:
                entities["brands"].append(brand)
        
        # Extract model patterns
        for category_data in self.EQUIPMENT_TAXONOMY.values():
            for pattern in category_data.get("model_patterns", []):
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    entities["models"].extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
        
        # Extract common components
        all_components = []
        for category_data in self.EQUIPMENT_TAXONOMY.values():
            all_components.extend(category_data.get("components", []))
        
        for component in all_components:
            if component.lower() in query_lower:
                entities["components"].append(component)
        
        # Extract part numbers (alphanumeric codes)
        part_patterns = [
            r'\b[A-Z]{2,4}[-\s]*\d{3,8}[A-Z]*\b',  # Standard part codes
            r'\bpart[:\s#]*([A-Z0-9\-]{4,12})\b',   # Part number references
            r'\bmodel[:\s]*([A-Z0-9\-]{3,12})\b'    # Model number references
        ]
        
        for pattern in part_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["part_numbers"].extend(matches)
        
        # Remove duplicates and empty strings
        for key in entities:
            entities[key] = list(set(filter(None, entities[key])))
        
        return entities
    
    def _create_taxonomy_summary(self) -> str:
        """Create a concise summary of the equipment taxonomy"""
        summary_lines = []
        
        for category, data in self.EQUIPMENT_TAXONOMY.items():
            brands = ", ".join(data["common_brands"][:8])  # First 8 brands
            components = ", ".join(data["components"][:8])  # First 8 components
            summary_lines.append(
                f"• {category.upper()}: Brands: {brands}... | Components: {components}..."
            )
        
        return "\n".join(summary_lines)
    
    def _validate_and_enhance_classification(
        self, 
        result: Dict[str, Any], 
        query: str, 
        extracted_entities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Validate and enhance the classification result"""
        
        # Ensure category is valid
        if result.get("category") not in self.EQUIPMENT_TAXONOMY:
            # Try to infer from extracted entities
            result["category"] = self._infer_category_from_entities(extracted_entities, query)
        
        category = result["category"]
        category_data = self.EQUIPMENT_TAXONOMY.get(category, {})
        
        # Validate subcategory
        valid_subcategories = category_data.get("subcategories", [])
        if result.get("subcategory") not in valid_subcategories:
            result["subcategory"] = valid_subcategories[0] if valid_subcategories else "unknown"
        
        # Enhance with extracted entities if missing
        if not result.get("brand") and extracted_entities.get("brands"):
            result["brand"] = extracted_entities["brands"][0]
        
        if not result.get("model") and extracted_entities.get("models"):
            result["model"] = extracted_entities["models"][0]
        
        if not result.get("year") and extracted_entities.get("years"):
            result["year"] = extracted_entities["years"][0]
        
        if not result.get("component") and extracted_entities.get("components"):
            result["component"] = extracted_entities["components"][0]
        
        # Ensure confidence is in valid range
        confidence = result.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            confidence = 0.7
        result["confidence"] = confidence
        
        # Set default values for missing fields
        result.setdefault("extracted_entities", extracted_entities)
        result.setdefault("search_keywords", self._generate_search_keywords(result, query))
        result.setdefault("priority_sources", category_data.get("priority_sources", ["general_manuals"]))
        result.setdefault("reasoning", "Classified based on query analysis and taxonomy matching")
        result.setdefault("alternative_interpretations", [])
        result.setdefault("task", "unknown")
        
        # Add metadata
        result["original_query"] = query
        result["category_confidence"] = self._calculate_category_confidence(result, extracted_entities)
        
        return result
    
    def _infer_category_from_entities(self, entities: Dict[str, List[str]], query: str) -> str:
        """Infer equipment category from extracted entities"""
        
        query_lower = query.lower()
        category_scores = {}
        
        # Score based on brand matches
        for category, data in self.EQUIPMENT_TAXONOMY.items():
            score = 0
            
            # Brand matches
            brand_matches = [brand for brand in entities.get("brands", []) 
                           if brand.lower() in [b.lower() for b in data.get("common_brands", [])]]
            score += len(brand_matches) * 3
            
            # Component matches
            component_matches = [comp for comp in entities.get("components", [])
                               if comp.lower() in [c.lower() for c in data.get("components", [])]]
            score += len(component_matches) * 2
            
            # Keyword matches in query
            category_keywords = data.get("common_brands", []) + data.get("components", [])
            keyword_matches = [kw for kw in category_keywords if kw.lower() in query_lower]
            score += len(keyword_matches)
            
            category_scores[category] = score
        
        # Return category with highest score, or default
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return "automotive"  # Default fallback
    
    def _generate_search_keywords(self, result: Dict[str, Any], query: str) -> List[str]:
        """Generate prioritized search keywords"""
        
        keywords = []
        
        # Add specific identifiers
        if result.get("year"):
            keywords.append(result["year"])
        if result.get("brand"):
            keywords.append(result["brand"])
        if result.get("model"):
            keywords.append(result["model"])
        if result.get("component"):
            keywords.append(result["component"])
        if result.get("task"):
            keywords.append(result["task"])
        
        # Add category-specific terms
        keywords.append(result.get("category", ""))
        keywords.append(result.get("subcategory", ""))
        
        # Add extracted entities
        entities = result.get("extracted_entities", {})
        for entity_list in entities.values():
            keywords.extend(entity_list)
        
        # Add query terms (filtered)
        query_words = [word.lower() for word in query.split() 
                      if len(word) > 3 and word.lower() not in ["the", "and", "for", "with"]]
        keywords.extend(query_words)
        
        # Remove duplicates and empty strings, maintain order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword.lower() not in seen:
                seen.add(keyword.lower())
                unique_keywords.append(keyword)
        
        return unique_keywords[:15]  # Limit to top 15 keywords
    
    def _calculate_category_confidence(self, result: Dict[str, Any], entities: Dict[str, List[str]]) -> float:
        """Calculate confidence score for category classification"""
        
        category = result.get("category", "")
        if category not in self.EQUIPMENT_TAXONOMY:
            return 0.3
        
        category_data = self.EQUIPMENT_TAXONOMY[category]
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on matches
        if result.get("brand") and result["brand"].lower() in [b.lower() for b in category_data.get("common_brands", [])]:
            confidence += 0.2
        
        if result.get("component") and result["component"].lower() in [c.lower() for c in category_data.get("components", [])]:
            confidence += 0.2
        
        if entities.get("brands") or entities.get("models"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _fallback_equipment_classification(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Rule-based fallback classification when LLM fails"""
        
        query_lower = query.lower()
        
        # Determine category based on keywords and entities
        category = self._infer_category_from_entities(entities, query)
        category_data = self.EQUIPMENT_TAXONOMY.get(category, {})
        
        # Extract basic information
        brand = entities.get("brands", [None])[0]
        model = entities.get("models", [None])[0] 
        year = entities.get("years", [None])[0]
        component = entities.get("components", [None])[0]
        
        # Infer task from query
        task_keywords = {
            "replace": "replacement",
            "repair": "repair", 
            "fix": "repair",
            "install": "installation",
            "maintain": "maintenance",
            "service": "maintenance",
            "troubleshoot": "troubleshooting",
            "diagnose": "diagnosis"
        }
        
        task = "unknown"
        for keyword, task_type in task_keywords.items():
            if keyword in query_lower:
                task = task_type
                break
        
        confidence = 0.6
        if brand and component:
            confidence = 0.8
        elif brand or component:
            confidence = 0.7
        
        return {
            "category": category,
            "subcategory": category_data.get("subcategories", ["unknown"])[0],
            "brand": brand,
            "model": model,
            "year": year,
            "component": component,
            "task": task,
            "confidence": confidence,
            "extracted_entities": entities,
            "search_keywords": self._generate_search_keywords({
                "category": category,
                "brand": brand,
                "model": model,
                "year": year,
                "component": component,
                "task": task
            }, query),
            "priority_sources": category_data.get("priority_sources", ["general_manuals"]),
            "reasoning": "Fallback rule-based classification using entity extraction",
            "alternative_interpretations": [],
            "original_query": query,
            "category_confidence": self._calculate_category_confidence({
                "category": category,
                "brand": brand,
                "component": component
            }, entities)
        }
    
    def get_category_description(self, category: str) -> str:
        """Get human-readable description of an equipment category"""
        descriptions = {
            "automotive": "Vehicles and automotive systems",
            "appliances": "Home and commercial appliances",
            "industrial": "Industrial equipment and machinery",
            "electronics": "Electronic devices and systems",
            "hvac": "Heating, ventilation, and air conditioning"
        }
        return descriptions.get(category, "Unknown equipment category")
    
    def get_search_strategy(self, equipment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate search strategy based on equipment classification"""
        
        category = equipment_result.get("category", "")
        brand = equipment_result.get("brand", "")
        model = equipment_result.get("model", "")
        
        strategy = {
            "primary_keywords": equipment_result.get("search_keywords", [])[:5],
            "secondary_keywords": equipment_result.get("search_keywords", [])[5:10],
            "content_type_priority": ["text", "diagrams", "tables"],
            "source_priority": equipment_result.get("priority_sources", []),
            "search_scope": "local_first"
        }
        
        # Adjust strategy based on classification
        if brand and model:
            strategy["search_scope"] = "external_preferred"
            strategy["primary_keywords"] = [brand, model] + strategy["primary_keywords"][:3]
        
        if equipment_result.get("task") in ["installation", "repair", "troubleshooting"]:
            strategy["content_type_priority"] = ["diagrams", "text", "tables"]
        elif equipment_result.get("task") == "specifications":
            strategy["content_type_priority"] = ["tables", "text", "diagrams"]
        
        return strategy
