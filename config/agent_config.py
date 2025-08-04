#!/usr/bin/env python3
"""
Agent Configuration Loader
Provides configuration for the agent pipeline system
"""

import os
from models.agent_models import (
    PipelineConfig, 
    IntentRouterConfig, 
    EquipmentClassifierConfig
)


def load_agent_config() -> PipelineConfig:
    """Load agent configuration from environment variables"""
    
    # Intent Router Configuration
    intent_router_config = IntentRouterConfig(
        model=os.getenv('INTENT_ROUTER_MODEL', 'gpt-4o-mini'),
        temperature=float(os.getenv('INTENT_ROUTER_TEMP', '0.1')),
        max_tokens=int(os.getenv('INTENT_ROUTER_MAX_TOKENS', '500')),
        timeout=float(os.getenv('INTENT_ROUTER_TIMEOUT', '30.0')),
        confidence_threshold=float(os.getenv('INTENT_CONFIDENCE_THRESHOLD', '0.8')),
        enable_fallback_classification=os.getenv('INTENT_FALLBACK_ENABLED', 'true').lower() == 'true'
    )
    
    # Equipment Classifier Configuration
    equipment_classifier_config = EquipmentClassifierConfig(
        model=os.getenv('EQUIPMENT_CLASSIFIER_MODEL', 'gpt-4o-mini'),
        temperature=float(os.getenv('EQUIPMENT_CLASSIFIER_TEMP', '0.1')),
        max_tokens=int(os.getenv('EQUIPMENT_CLASSIFIER_MAX_TOKENS', '600')),
        timeout=float(os.getenv('EQUIPMENT_CLASSIFIER_TIMEOUT', '30.0')),
        enable_entity_extraction=os.getenv('EQUIPMENT_ENTITY_EXTRACTION', 'true').lower() == 'true',
        confidence_threshold=float(os.getenv('EQUIPMENT_CONFIDENCE_THRESHOLD', '0.7')),
        max_search_keywords=int(os.getenv('MAX_SEARCH_KEYWORDS', '15'))
    )
    
    # Complete Pipeline Configuration
    pipeline_config = PipelineConfig(
        intent_router=intent_router_config,
        equipment_classifier=equipment_classifier_config,
        enable_caching=os.getenv('AGENT_CACHING_ENABLED', 'true').lower() == 'true',
        cache_ttl_seconds=int(os.getenv('AGENT_CACHE_TTL', '300')),
        max_concurrent_agents=int(os.getenv('MAX_CONCURRENT_AGENTS', '5'))
    )
    
    return pipeline_config
