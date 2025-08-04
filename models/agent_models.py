#!/usr/bin/env python3
"""
Pydantic Models for Agent System
Type-safe data models for all agent interactions
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class IntentCategory(str, Enum):
    """Enumeration of available intent categories"""
    LOCAL_SEARCH_ONLY = "local_search_only"
    MANUAL_SEARCH_NEEDED = "manual_search_needed"
    GENERAL_KNOWLEDGE = "general_knowledge"
    EQUIPMENT_SPECIFIC = "equipment_specific"
    TROUBLESHOOTING = "troubleshooting"
    MAINTENANCE = "maintenance"
    SPECIFICATIONS = "specifications"
    PARTS_DIAGRAM = "parts_diagram"
    WIRING_ELECTRICAL = "wiring_electrical"
    INSTALLATION = "installation"
    DIAGRAM_REQUEST = "diagram_request"  # Added missing enum value
    GENERAL_INQUIRY = "general_inquiry"  # Added missing enum value


class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class EquipmentCategory(str, Enum):
    """Equipment category enumeration"""
    AUTOMOTIVE = "automotive"
    APPLIANCES = "appliances"
    APPLIANCE = "appliance"  # Added for consistency
    INDUSTRIAL = "industrial"
    ELECTRONICS = "electronics"
    HVAC = "hvac"
    UNKNOWN = "unknown"  # Added missing enum value


class SearchScope(str, Enum):
    """Search scope options"""
    LOCAL_FIRST = "local_first"
    EXTERNAL_PREFERRED = "external_preferred"
    LOCAL_ONLY = "local_only"
    EXTERNAL_ONLY = "external_only"
    HYBRID = "hybrid"  # Added missing enum value


class ContentType(str, Enum):
    """Content type enumeration"""
    TEXT = "text"
    TABLES = "tables"
    DIAGRAMS = "diagrams"
    FIGURES = "figures"
    VIDEOS = "videos"  # Added missing enum value


class IntentAnalysis(BaseModel):
    """Intent analysis result model"""
    primary_intent: IntentCategory
    secondary_intents: List[IntentCategory] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    requires_manual: bool = False
    requires_equipment_classification: bool = False
    query_complexity: QueryComplexity = QueryComplexity.MODERATE
    reasoning: str
    keywords_detected: List[str] = Field(default_factory=list)
    content_types_needed: List[ContentType] = Field(default_factory=list)
    original_query: str
    query_length: int = Field(ge=0)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    @validator('secondary_intents')
    def validate_secondary_intents(cls, v, values):
        primary = values.get('primary_intent')
        if primary and primary in v:
            raise ValueError('Primary intent cannot be in secondary intents')
        return v


class ExtractedEntities(BaseModel):
    """Extracted entities from query"""
    years: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    components: List[str] = Field(default_factory=list)
    part_numbers: List[str] = Field(default_factory=list)


class EquipmentAnalysis(BaseModel):
    """Equipment analysis result model"""
    category: EquipmentCategory
    subcategory: str
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[str] = None
    component: Optional[str] = None
    task: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_entities: ExtractedEntities
    search_keywords: List[str] = Field(default_factory=list)
    priority_sources: List[str] = Field(default_factory=list)
    reasoning: str
    alternative_interpretations: List[str] = Field(default_factory=list)
    original_query: str
    category_confidence: float = Field(ge=0.0, le=1.0)
    
    @validator('confidence', 'category_confidence')
    def validate_confidence_fields(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence values must be between 0.0 and 1.0')
        return v


class SearchMetadata(BaseModel):
    """Search strategy metadata"""
    intent_primary: IntentCategory
    intent_confidence: float = Field(ge=0.0, le=1.0)
    equipment_category: EquipmentCategory
    equipment_confidence: float = Field(ge=0.0, le=1.0)
    has_specific_equipment: bool = False
    query_complexity: QueryComplexity = QueryComplexity.MODERATE


class SearchStrategy(BaseModel):
    """Search strategy configuration"""
    search_scope: SearchScope = SearchScope.LOCAL_FIRST
    content_type_priority: List[ContentType] = Field(default_factory=list)
    primary_keywords: List[str] = Field(default_factory=list)
    secondary_keywords: List[str] = Field(default_factory=list)
    source_priority: List[str] = Field(default_factory=list)
    needs_external_search: bool = False
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results_per_type: int = Field(default=5, ge=1, le=20)
    enable_hybrid_search: bool = True
    search_metadata: SearchMetadata
    
    @validator('max_results_per_type')
    def validate_max_results(cls, v):
        if not 1 <= v <= 20:
            raise ValueError('max_results_per_type must be between 1 and 20')
        return v


class AgentStepSummary(BaseModel):
    """Summary of an agent processing step"""
    agent: str
    duration: float = Field(ge=0.0)
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    error: Optional[str] = None


class PipelineMetadata(BaseModel):
    """Pipeline processing metadata"""
    intent_router_used: bool = False
    equipment_classifier_used: bool = False
    total_agents_used: int = Field(ge=0)
    fastest_step: Optional[str] = None
    slowest_step: Optional[str] = None
    overall_confidence: float = Field(ge=0.0, le=1.0)


class PipelineResult(BaseModel):
    """Complete pipeline processing result"""
    success: bool
    query: str
    processing_time: float = Field(ge=0.0)
    steps: List[AgentStepSummary]
    agent_count: int = Field(ge=0)
    pipeline_version: str = "1.0"
    intent_analysis: Optional[IntentAnalysis] = None
    equipment_analysis: Optional[EquipmentAnalysis] = None
    search_strategy: Optional[SearchStrategy] = None
    error: Optional[str] = None
    metadata: PipelineMetadata
    
    @validator('agent_count')
    def validate_agent_count(cls, v, values):
        steps = values.get('steps', [])
        if v != len(steps):
            raise ValueError('agent_count must match the number of steps')
        return v


class QueryValidation(BaseModel):
    """Query validation result"""
    valid: bool
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AgentStatus(BaseModel):
    """Agent status information"""
    agent_name: str
    config: Dict[str, Any]
    initialized: bool = True
    last_updated: str
    ready: bool = True
    error: Optional[str] = None


class PipelineStatus(BaseModel):
    """Overall pipeline status"""
    pipeline_ready: bool
    agents: Dict[str, AgentStatus]
    last_updated: float
    
    
class AgentResponseModel(BaseModel):
    """Standardized agent response model"""
    success: bool
    data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    processing_time: float = Field(ge=0.0)
    agent_name: str
    timestamp: str
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError('timestamp must be in ISO format')
        return v


class AgentConfig(BaseModel):
    """Agent configuration model"""
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=50, le=4000)
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v


class IntentRouterConfig(AgentConfig):
    """Intent router specific configuration"""
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_fallback_classification: bool = True


class EquipmentClassifierConfig(AgentConfig):
    """Equipment classifier specific configuration"""
    enable_entity_extraction: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_search_keywords: int = Field(default=15, ge=5, le=30)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration"""
    intent_router: IntentRouterConfig = Field(default_factory=IntentRouterConfig)
    equipment_classifier: EquipmentClassifierConfig = Field(default_factory=EquipmentClassifierConfig)
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    max_concurrent_agents: int = Field(default=5, ge=1, le=10)
    
    
class ProcessingStepInput(BaseModel):
    """Input data for pipeline processing steps"""
    query: str = Field(min_length=1, max_length=1000)
    context: str = ""
    intent_data: Optional[Dict[str, Any]] = None
    equipment_data: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def validate_query(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Query must be at least 5 characters long')
        return v.strip()


# Export all models for easy importing
__all__ = [
    'IntentCategory',
    'QueryComplexity', 
    'EquipmentCategory',
    'SearchScope',
    'ContentType',
    'IntentAnalysis',
    'ExtractedEntities',
    'EquipmentAnalysis',
    'SearchMetadata',
    'SearchStrategy',
    'AgentStepSummary',
    'PipelineMetadata',
    'PipelineResult',
    'QueryValidation',
    'AgentStatus',
    'PipelineStatus',
    'AgentResponseModel',
    'AgentConfig',
    'IntentRouterConfig',
    'EquipmentClassifierConfig',
    'PipelineConfig',
    'ProcessingStepInput'
]
