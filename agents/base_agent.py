#!/usr/bin/env python3
"""
Base Agent Class for Multimodal RAG Assistant
Abstract foundation for all intelligent agents in the system
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from openai import OpenAI


@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    success: bool
    data: Dict[str, Any]
    confidence: float
    reasoning: str
    processing_time: float
    agent_name: str
    timestamp: str
    errors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)


class BaseAgent(ABC):
    """Abstract base class for all intelligent agents"""
    
    def __init__(self, openai_client: OpenAI, config: Optional[Dict[str, Any]] = None):
        self.openai_client = openai_client
        self.config = config or {}
        self.logger = self._setup_logging()
        self.agent_name = self.__class__.__name__
        
        # Default configuration
        self.default_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 500,
            "timeout": 30.0
        }
        
        # Merge with provided config
        self.effective_config = {**self.default_config, **self.config}
        
        self.logger.info(f"Initialized {self.agent_name} with config: {self.effective_config}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up agent-specific logging"""
        logger = logging.getLogger(f"agents.{self.__class__.__name__.lower()}")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Only add handler if it doesn't exist
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Main processing method - must be implemented by subclasses
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            AgentResponse: Standardized response object
        """
        pass
    
    async def _call_openai(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Helper method to call OpenAI API with error handling
        
        Args:
            messages: List of message dictionaries for the conversation
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: Response content from OpenAI
        """
        try:
            # Merge with effective config
            call_config = {**self.effective_config, **kwargs}
            
            response = self.openai_client.chat.completions.create(
                model=call_config["model"],
                messages=messages,
                temperature=call_config["temperature"],
                max_tokens=call_config["max_tokens"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _create_response(
        self, 
        success: bool, 
        data: Dict[str, Any], 
        confidence: float,
        reasoning: str,
        processing_time: float,
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Create a standardized agent response
        
        Args:
            success: Whether the processing was successful
            data: The main response data
            confidence: Confidence score (0.0 to 1.0)
            reasoning: Explanation of the agent's decision process
            processing_time: Time taken to process (seconds)
            errors: List of error messages (if any)
            metadata: Additional metadata
            
        Returns:
            AgentResponse: Standardized response object
        """
        return AgentResponse(
            success=success,
            data=data,
            confidence=confidence,
            reasoning=reasoning,
            processing_time=processing_time,
            agent_name=self.agent_name,
            timestamp=datetime.now().isoformat(),
            errors=errors,
            metadata=metadata
        )
    
    def use_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper for MCP tool usage - placeholder for future MCP integration
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to use
            arguments: Arguments for the tool
            
        Returns:
            Dict containing tool response
        """
        self.logger.info(f"MCP tool call: {server_name}.{tool_name} with args: {arguments}")
        
        # Placeholder implementation
        # In the future, this would integrate with actual MCP servers
        return {
            "success": False,
            "message": "MCP integration not yet implemented",
            "server": server_name,
            "tool": tool_name,
            "arguments": arguments
        }
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """
        Validate input data contains required keys
        
        Args:
            input_data: Input dictionary to validate
            required_keys: List of required keys
            
        Returns:
            List of missing keys (empty if validation passes)
        """
        missing_keys = []
        for key in required_keys:
            if key not in input_data or input_data[key] is None:
                missing_keys.append(key)
        
        if missing_keys:
            self.logger.warning(f"Missing required keys: {missing_keys}")
        
        return missing_keys
    
    def extract_confidence_score(self, response_text: str) -> float:
        """
        Extract confidence score from LLM response text
        
        Args:
            response_text: Text response from LLM
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        import re
        
        # Look for confidence patterns in the response
        confidence_patterns = [
            r'confidence[:\s]*([0-9]*\.?[0-9]+)',
            r'certainty[:\s]*([0-9]*\.?[0-9]+)',
            r'score[:\s]*([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Normalize to 0-1 range if needed
                    if score > 1.0:
                        score = score / 100.0
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
        
        # Default confidence if no pattern found
        return 0.7
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and configuration
        
        Returns:
            Dict containing agent status information
        """
        return {
            "agent_name": self.agent_name,
            "config": self.effective_config,
            "initialized": True,
            "last_updated": datetime.now().isoformat()
        }


class AgentError(Exception):
    """Custom exception for agent-related errors"""
    
    def __init__(self, message: str, agent_name: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.agent_name = agent_name
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "error_type": "AgentError",
            "message": self.message,
            "agent_name": self.agent_name,
            "details": self.details,
            "timestamp": datetime.now().isoformat()
        }
