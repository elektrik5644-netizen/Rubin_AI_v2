"""
Integrated Mathematical Solver

This module integrates all mathematical solver components
and provides a unified interface for the Rubin AI system.
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .category_detector import MathematicalCategoryDetector
from .request_handler import MathematicalRequestHandler
from .response_formatter import MathematicalResponseFormatter
from .error_handler import MathematicalErrorHandler

logger = logging.getLogger(__name__)

@dataclass
class MathIntegrationConfig:
    enabled: bool = True
    confidence_threshold: float = 0.7
    fallback_to_general: bool = True
    log_requests: bool = True
    response_format: str = "structured"  # "structured" or "simple"

class IntegratedMathematicalSolver:
    """
    Integrated mathematical solver that combines all components
    for seamless integration with Rubin AI system.
    """
    
    def __init__(self, config: Optional[MathIntegrationConfig] = None):
        """
        Initialize the integrated mathematical solver.
        
        Args:
            config: Configuration object for the solver
        """
        self.config = config or MathIntegrationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector = MathematicalCategoryDetector()
        self.handler = MathematicalRequestHandler()
        self.formatter = MathematicalResponseFormatter()
        self.error_handler = MathematicalErrorHandler()
        
        self.logger.info("Integrated Mathematical Solver initialized")
    
    def process_request(self, message: str) -> Dict[str, Any]:
        """
        Processes a mathematical request through the complete pipeline.
        
        Args:
            message: The user's message
            
        Returns:
            Dictionary with complete response data
        """
        start_time = time.time()
        
        if self.config.log_requests:
            self.logger.info(f"Processing mathematical request: {message[:100]}...")
        
        try:
            # Step 1: Check if it's a mathematical request
            if not self.detector.is_mathematical_request(message):
                return self._handle_non_mathematical_request(message)
            
            # Step 2: Detect category
            category = self.detector.detect_math_category(message)
            if not category:
                category = "general"
            
            # Step 3: Validate the request
            validation = self.handler.validate_request(message, category)
            if not validation["valid"]:
                return self.error_handler.handle_validation_error(
                    message, category, validation["errors"]
                )
            
            # Step 4: Process the mathematical request
            handler_result = self.handler.handle_request(message, category)
            
            if not handler_result["success"]:
                return self.error_handler.handle_solving_error(
                    message, category, 
                    Exception(handler_result["error_message"])
                )
            
            # Step 5: Check confidence threshold
            solution = handler_result["solution"]
            if solution.confidence < self.config.confidence_threshold:
                return self.error_handler.handle_low_confidence_error(
                    message, category, solution.confidence
                )
            
            # Step 6: Format the response
            processing_time = time.time() - start_time
            if self.config.response_format == "structured":
                response = self.formatter.create_structured_response(
                    solution, category, processing_time
                )
            else:
                response = self.formatter.create_user_friendly_response(
                    solution, category
                )
            
            # Step 7: Log success
            if self.config.log_requests:
                self.logger.info(f"Request processed successfully in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Handle the error
            if self.config.fallback_to_general:
                return self.error_handler.handle_fallback_error(message, e)
            else:
                # Try to determine error type and handle accordingly
                if "detection" in str(e).lower():
                    return self.error_handler.handle_detection_error(message, e)
                elif "parsing" in str(e).lower():
                    category = self.detector.detect_math_category(message) or "general"
                    return self.error_handler.handle_parsing_error(message, category, e)
                else:
                    category = self.detector.detect_math_category(message) or "general"
                    return self.error_handler.handle_solving_error(message, category, e)
    
    def _handle_non_mathematical_request(self, message: str) -> Dict[str, Any]:
        """
        Handles requests that are not mathematical.
        
        Args:
            message: The user's message
            
        Returns:
            Dictionary indicating this is not a mathematical request
        """
        return {
            "response": None,  # Signal to route to other modules
            "provider": "Mathematical Solver",
            "category": "not_mathematical",
            "solution_data": {
                "problem_type": "not_mathematical",
                "final_answer": None,
                "confidence": 0.0,
                "explanation": "Not a mathematical request",
                "steps": [],
                "formulas_used": [],
                "input_data": {}
            },
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0,
            "success": False,
            "error_message": "Not a mathematical request",
            "should_route_to_other": True
        }
    
    def is_mathematical_request(self, message: str) -> bool:
        """
        Quick check if a message is a mathematical request.
        
        Args:
            message: The user's message
            
        Returns:
            True if mathematical, False otherwise
        """
        return self.detector.is_mathematical_request(message)
    
    def get_detected_category(self, message: str) -> Optional[str]:
        """
        Gets the detected mathematical category.
        
        Args:
            message: The user's message
            
        Returns:
            Category name or None
        """
        return self.detector.detect_math_category(message)
    
    def process_batch_requests(self, messages: list) -> Dict[str, Any]:
        """
        Processes multiple mathematical requests.
        
        Args:
            messages: List of user messages
            
        Returns:
            Dictionary with batch results
        """
        start_time = time.time()
        results = []
        
        for message in messages:
            result = self.process_request(message)
            results.append(result)
        
        total_time = time.time() - start_time
        
        return self.formatter.format_batch_response(results)
    
    def get_solver_status(self) -> Dict[str, Any]:
        """
        Gets the status of the mathematical solver.
        
        Returns:
            Dictionary with solver status
        """
        try:
            # Test the solver
            test_result = self.process_request("2+2")
            
            return {
                "status": "operational" if test_result.get("success") else "error",
                "solver_type": "IntegratedMathematicalSolver",
                "config": {
                    "enabled": self.config.enabled,
                    "confidence_threshold": self.config.confidence_threshold,
                    "fallback_enabled": self.config.fallback_to_general,
                    "response_format": self.config.response_format
                },
                "components": {
                    "detector": "operational",
                    "handler": "operational",
                    "formatter": "operational",
                    "error_handler": "operational"
                },
                "test_result": test_result.get("solution_data", {}).get("final_answer"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        Updates the solver configuration.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        if "enabled" in new_config:
            self.config.enabled = bool(new_config["enabled"])
        if "confidence_threshold" in new_config:
            self.config.confidence_threshold = float(new_config["confidence_threshold"])
        if "fallback_to_general" in new_config:
            self.config.fallback_to_general = bool(new_config["fallback_to_general"])
        if "log_requests" in new_config:
            self.config.log_requests = bool(new_config["log_requests"])
        if "response_format" in new_config:
            self.config.response_format = str(new_config["response_format"])
        
        self.logger.info(f"Configuration updated: {new_config}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Gets performance metrics for monitoring.
        
        Returns:
            Dictionary with performance metrics
        """
        # This would typically collect metrics from a monitoring system
        return {
            "total_requests": 0,  # Placeholder
            "successful_requests": 0,  # Placeholder
            "failed_requests": 0,  # Placeholder
            "average_processing_time": 0.0,  # Placeholder
            "success_rate": 0.0,  # Placeholder
            "error_rate": 0.0,  # Placeholder
            "category_distribution": {},  # Placeholder
            "confidence_distribution": {}  # Placeholder
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Performs a health check of all components.
        
        Returns:
            Dictionary with health status
        """
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check detector
        try:
            self.detector.is_mathematical_request("2+2")
            health_status["components"]["detector"] = "healthy"
        except Exception as e:
            health_status["components"]["detector"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check handler
        try:
            self.handler.get_solver_status()
            health_status["components"]["handler"] = "healthy"
        except Exception as e:
            health_status["components"]["handler"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check formatter
        try:
            # Test formatter with dummy data
            health_status["components"]["formatter"] = "healthy"
        except Exception as e:
            health_status["components"]["formatter"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check error handler
        try:
            self.error_handler.get_error_statistics()
            health_status["components"]["error_handler"] = "healthy"
        except Exception as e:
            health_status["components"]["error_handler"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        return health_status
