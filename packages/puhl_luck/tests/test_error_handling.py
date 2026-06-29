"""
Tests for comprehensive error handling (Task 15.1).
"""

import pytest
import logging
from puhl_luck._memory_error_handling import (
    FieldError, FieldFormationError, EnergyComputationError,
    InputValidationError, PersistenceError,
    handle_field_errors, validate_input,
    validate_non_empty_string, validate_positive_number, validate_in_range,
    ErrorRecovery, FieldOperationContext,
    configure_logging
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""
    
    def test_field_error_base(self):
        """Test FieldError base exception."""
        with pytest.raises(FieldError):
            raise FieldError("Test error")
    
    def test_specific_exceptions(self):
        """Test specific exception types."""
        with pytest.raises(FieldFormationError):
            raise FieldFormationError("Field formation failed")
        
        with pytest.raises(EnergyComputationError):
            raise EnergyComputationError("Energy computation failed")
        
        with pytest.raises(InputValidationError):
            raise InputValidationError("Invalid input")


class TestErrorHandlingDecorators:
    """Test error handling decorators."""
    
    def test_handle_field_errors_catches_errors(self):
        """Test that decorator catches and handles errors."""
        
        @handle_field_errors(FieldError, default_return="fallback")
        def failing_function():
            raise FieldError("Test error")
        
        result = failing_function()
        assert result == "fallback"
    
    def test_handle_field_errors_allows_success(self):
        """Test that decorator allows successful execution."""
        
        @handle_field_errors(FieldError, default_return="fallback")
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_validate_input_decorator(self):
        """Test input validation decorator."""
        
        @validate_input(
            lambda x: x > 0,
            "Value must be positive"
        )
        def requires_positive(value):
            return value * 2
        
        # Valid input
        assert requires_positive(5) == 10
        
        # Invalid input
        with pytest.raises(InputValidationError):
            requires_positive(-1)


class TestValidationFunctions:
    """Test validation utility functions."""
    
    def test_validate_non_empty_string(self):
        """Test string validation."""
        # Valid strings
        validate_non_empty_string("hello")
        validate_non_empty_string("  test  ")
        
        # Invalid strings
        with pytest.raises(InputValidationError):
            validate_non_empty_string("")
        
        with pytest.raises(InputValidationError):
            validate_non_empty_string("   ")
        
        with pytest.raises(InputValidationError):
            validate_non_empty_string(None)
    
    def test_validate_positive_number(self):
        """Test positive number validation."""
        # Valid numbers
        validate_positive_number(1)
        validate_positive_number(0.5)
        validate_positive_number(100.0)
        
        # Invalid numbers
        with pytest.raises(InputValidationError):
            validate_positive_number(0)
        
        with pytest.raises(InputValidationError):
            validate_positive_number(-1)
    
    def test_validate_in_range(self):
        """Test range validation."""
        # Valid values
        validate_in_range(0.5, 0.0, 1.0)
        validate_in_range(10, 0, 100)
        validate_in_range(0.0, 0.0, 1.0)  # Boundary
        validate_in_range(1.0, 0.0, 1.0)  # Boundary
        
        # Invalid values
        with pytest.raises(InputValidationError):
            validate_in_range(1.5, 0.0, 1.0)
        
        with pytest.raises(InputValidationError):
            validate_in_range(-0.1, 0.0, 1.0)


class TestErrorRecovery:
    """Test error recovery utilities."""
    
    def test_safe_computation_success(self):
        """Test safe computation with successful operation."""
        result = ErrorRecovery.safe_computation(
            lambda: 2 + 2,
            fallback_value=0
        )
        assert result == 4
    
    def test_safe_computation_fallback(self):
        """Test safe computation with error."""
        result = ErrorRecovery.safe_computation(
            lambda: 1 / 0,  # Will raise ZeroDivisionError
            fallback_value=0,
            error_name="division"
        )
        assert result == 0
    
    def test_safe_computation_nan_detection(self):
        """Test that NaN values trigger fallback."""
        result = ErrorRecovery.safe_computation(
            lambda: float('nan'),
            fallback_value=0.0
        )
        assert result == 0.0
    
    def test_safe_list_operation_success(self):
        """Test safe list operation with success."""
        result = ErrorRecovery.safe_list_operation(
            lambda: [1, 2, 3]
        )
        assert result == [1, 2, 3]
    
    def test_safe_list_operation_fallback(self):
        """Test safe list operation with error."""
        result = ErrorRecovery.safe_list_operation(
            lambda: [][10],  # Index error
            empty_list_fallback=True
        )
        assert result == []
    
    def test_check_oscillation_detects(self):
        """Test oscillation detection."""
        # Oscillating sequence
        oscillating = [1.0, 0.5, 1.0, 0.5, 1.0, 0.5]
        assert ErrorRecovery.check_oscillation(oscillating, window=5)
        
        # Non-oscillating sequence
        stable = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        assert not ErrorRecovery.check_oscillation(stable, window=5)
    
    def test_check_convergence_detects(self):
        """Test convergence detection."""
        # Converged sequence
        converged = [1.0, 1.001, 1.002, 1.003]
        assert ErrorRecovery.check_convergence(converged, threshold=0.01, window=3)
        
        # Not converged sequence
        diverging = [1.0, 1.5, 2.0, 2.5]
        assert not ErrorRecovery.check_convergence(diverging, threshold=0.01, window=3)


class TestFieldOperationContext:
    """Test context manager for field operations."""
    
    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        with FieldOperationContext("test_operation") as ctx:
            result = 2 + 2
        
        assert ctx.success is True
    
    def test_context_manager_handles_error(self):
        """Test context manager handles errors gracefully."""
        with FieldOperationContext("test_operation", critical=False) as ctx:
            raise ValueError("Test error")
        
        assert ctx.success is False
    
    def test_context_manager_critical_reraises(self):
        """Test that critical errors are re-raised."""
        with pytest.raises(ValueError):
            with FieldOperationContext("test_operation", critical=True):
                raise ValueError("Critical error")


class TestLoggingConfiguration:
    """Test logging configuration."""
    
    def test_configure_logging_basic(self):
        """Test basic logging configuration."""
        configure_logging(level=logging.INFO)
        # If no exception, configuration succeeded
        assert True
    
    def test_logging_works_after_config(self):
        """Test that logging works after configuration."""
        from puhl_luck._memory_error_handling import logger
        
        # This should not raise an exception
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")


class TestIntegrationWithComponents:
    """Integration tests for error handling with components."""
    
    def test_field_formation_with_error_handling(self):
        """Test field formation handles errors gracefully."""
        from puhl_luck._memory_field_formation import FieldFormation
        from puhl_luck._memory_exposure_layer import ExposureEventsLayer
        from puhl_luck._memory_field_core import InputContext
        
        events_layer = ExposureEventsLayer()
        formation = FieldFormation()
        
        # Empty input should be handled gracefully
        try:
            context = InputContext.from_text("")
            field = formation.form_field(context, events_layer)
            # Should return a field even with empty input
            assert field is not None
        except InputValidationError:
            # Or raise validation error - both are acceptable
            pass
    
    def test_energy_computation_with_invalid_values(self):
        """Test energy computation handles invalid values."""
        from puhl_luck._memory_field_energy import FreeEnergyMinimization
        from puhl_luck._memory_field_core import StateField, FieldEnergy
        
        energy_calc = FreeEnergyMinimization()
        
        # Create field with empty activations
        field = StateField(
            query_features=[],
            query_hv=None,
            activated_events={},
            activated_concepts={},
            activated_operators={},
            conflict_markers=[],
            goal_states=[],
            partial_outputs=[]
        )
        
        # Should handle empty field gracefully (returns valid FieldEnergy)
        energy = energy_calc.compute_field_energy(field)
        assert energy is not None
        assert isinstance(energy, FieldEnergy)
        # Energy values should be finite
        assert not (energy.total != energy.total)  # Not NaN
        assert -1e10 < energy.total < 1e10  # Finite


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
