"""
Comprehensive Error Handling for Cognitive Field Architecture.

Provides error handling utilities, custom exceptions, and recovery mechanisms
for all field-based operations. Ensures graceful degradation and proper logging.

Task 15.1: Error handling for all components
"""

import logging
from typing import Any, Optional, Callable, Dict
from functools import wraps
import traceback

# Configure logger
logger = logging.getLogger(__name__)


# Custom Exceptions

class FieldError(Exception):
    """Base exception for all field-related errors."""
    pass


class FieldFormationError(FieldError):
    """Error during field formation process."""
    pass


class EnergyComputationError(FieldError):
    """Error during energy computation."""
    pass


class CandidateGenerationError(FieldError):
    """Error during candidate generation."""
    pass


class StabilizationError(FieldError):
    """Error during recursive stabilization."""
    pass


class PersistenceError(FieldError):
    """Error during save/load operations."""
    pass


class InputValidationError(FieldError):
    """Error from invalid input."""
    pass


class MemoryCapacityError(FieldError):
    """Error when memory capacity is exceeded."""
    pass


# Error Handling Decorators

def handle_field_errors(error_class=FieldError, default_return=None, log_level=logging.ERROR):
    """
    Decorator to handle and log field errors gracefully.
    
    Args:
        error_class: Exception class to catch
        default_return: Value to return on error
        log_level: Logging level for error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_class as e:
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                return default_return
            except Exception as e:
                logger.log(
                    logging.CRITICAL,
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def validate_input(validation_func: Callable[[Any], bool], error_msg: str):
    """
    Decorator to validate function inputs.
    
    Args:
        validation_func: Function that returns True if input is valid
        error_msg: Error message to raise if validation fails
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validation_func(*args, **kwargs):
                logger.error(f"Input validation failed for {func.__name__}: {error_msg}")
                raise InputValidationError(error_msg)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Validation Functions

def validate_non_empty_string(s: str, name: str = "input") -> None:
    """Validate that a string is not None or empty."""
    if not s or not isinstance(s, str) or not s.strip():
        raise InputValidationError(f"{name} must be a non-empty string")


def validate_positive_number(n: float, name: str = "value") -> None:
    """Validate that a number is positive."""
    if not isinstance(n, (int, float)) or n <= 0:
        raise InputValidationError(f"{name} must be a positive number, got {n}")


def validate_in_range(value: float, min_val: float, max_val: float, name: str = "value") -> None:
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise InputValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def validate_dict_not_empty(d: Dict, name: str = "dictionary") -> None:
    """Validate that a dictionary is not empty."""
    if not isinstance(d, dict) or len(d) == 0:
        raise InputValidationError(f"{name} must be a non-empty dictionary")


# Error Recovery Utilities

class ErrorRecovery:
    """Utilities for error recovery and graceful degradation."""
    
    @staticmethod
    def safe_computation(
        computation: Callable,
        fallback_value: Any,
        error_name: str = "computation"
    ) -> Any:
        """
        Safely execute a computation with fallback.
        
        Args:
            computation: Function to execute
            fallback_value: Value to return on error
            error_name: Name for logging
            
        Returns:
            Result of computation or fallback value
        """
        try:
            result = computation()
            
            # Check for NaN/inf in numeric results
            if isinstance(result, (int, float)):
                if not (-1e308 < result < 1e308):  # Rough inf check
                    logger.warning(f"{error_name} produced inf/extreme value: {result}, using fallback")
                    return fallback_value
                
                # Check NaN
                if result != result:  # NaN check
                    logger.warning(f"{error_name} produced NaN, using fallback")
                    return fallback_value
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Error in {error_name}: {str(e)}, using fallback value",
                exc_info=True
            )
            return fallback_value
    
    @staticmethod
    def safe_list_operation(
        operation: Callable,
        empty_list_fallback: bool = True,
        error_name: str = "list operation"
    ) -> list:
        """
        Safely execute a list operation.
        
        Args:
            operation: Function that returns a list
            empty_list_fallback: If True, return empty list on error
            error_name: Name for logging
            
        Returns:
            Result list or empty list on error
        """
        try:
            result = operation()
            
            if not isinstance(result, list):
                logger.warning(f"{error_name} did not return a list, converting")
                result = list(result) if result else []
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Error in {error_name}: {str(e)}, returning empty list",
                exc_info=True
            )
            return [] if empty_list_fallback else None
    
    @staticmethod
    def check_oscillation(values: list, window: int = 5) -> bool:
        """
        Check if a sequence of values is oscillating.
        
        Args:
            values: List of numeric values
            window: Window size for oscillation detection
            
        Returns:
            True if oscillation detected
        """
        if len(values) < window:
            return False
        
        try:
            recent = values[-window:]
            
            # Check for alternating pattern
            differences = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            sign_changes = sum(
                1 for i in range(len(differences)-1)
                if differences[i] * differences[i+1] < 0
            )
            
            # If most differences change sign, it's oscillating
            return sign_changes >= (len(differences) - 1) * 0.7
            
        except Exception as e:
            logger.warning(f"Error checking oscillation: {e}")
            return False
    
    @staticmethod
    def check_convergence(values: list, threshold: float = 0.01, window: int = 3) -> bool:
        """
        Check if a sequence has converged.
        
        Args:
            values: List of numeric values
            threshold: Convergence threshold
            window: Window size for convergence check
            
        Returns:
            True if converged
        """
        if len(values) < window:
            return False
        
        try:
            recent = values[-window:]
            
            # Check if recent values are close
            max_diff = max(recent) - min(recent)
            return max_diff < threshold
            
        except Exception as e:
            logger.warning(f"Error checking convergence: {e}")
            return False


# Context Managers for Error Handling

class FieldOperationContext:
    """Context manager for field operations with error handling."""
    
    def __init__(self, operation_name: str, critical: bool = False):
        """
        Initialize context manager.
        
        Args:
            operation_name: Name of the operation for logging
            critical: If True, re-raise exceptions
        """
        self.operation_name = operation_name
        self.critical = critical
        self.success = False
    
    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            logger.debug(f"Operation completed successfully: {self.operation_name}")
            return True
        
        logger.error(
            f"Operation failed: {self.operation_name}",
            exc_info=(exc_type, exc_val, exc_tb)
        )
        
        if self.critical:
            # Re-raise for critical operations
            return False
        
        # Suppress non-critical errors
        self.success = False
        return True


# Logging Configuration

def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Configure logging for the field architecture.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional file path for log output
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    logger.info("Logging configured for field architecture")


__all__ = [
    # Exceptions
    'FieldError',
    'FieldFormationError',
    'EnergyComputationError',
    'CandidateGenerationError',
    'StabilizationError',
    'PersistenceError',
    'InputValidationError',
    'MemoryCapacityError',
    
    # Decorators
    'handle_field_errors',
    'validate_input',
    
    # Validation
    'validate_non_empty_string',
    'validate_positive_number',
    'validate_in_range',
    'validate_dict_not_empty',
    
    # Recovery
    'ErrorRecovery',
    
    # Context managers
    'FieldOperationContext',
    
    # Configuration
    'configure_logging',
    
    # Logger
    'logger',
]
