import torch
import pytest
import sys
import os

# Add the current directory to path to import models
sys.path.append('.')

from models.bin import BiN


def test_initialization():
    """Test BiN model initialization"""
    num_features = 10
    seq_length = 20
    
    model = BiN(num_features, seq_length)
    
    # Check dimensions are stored correctly
    assert model.num_features == num_features
    assert model.seq_length == seq_length
    
    # Check parameter shapes
    assert model.feature_bias.shape == (seq_length, 1)
    assert model.feature_scale.shape == (seq_length, 1)
    assert model.temporal_bias.shape == (num_features, 1)
    assert model.temporal_scale.shape == (num_features, 1)
    assert model.feature_weight.shape == (1,)
    assert model.temporal_weight.shape == (1,)
    
    # Check initial values
    assert torch.allclose(model.feature_bias, torch.zeros_like(model.feature_bias))
    assert torch.allclose(model.temporal_bias, torch.zeros_like(model.temporal_bias))
    assert torch.allclose(model.feature_weight, torch.tensor([0.5]))
    assert torch.allclose(model.temporal_weight, torch.tensor([0.5]))


def test_forward_pass_basic():
    """Test basic forward pass with different input shapes"""
    model = BiN(num_features=4, seq_length=6)
    
    # Test with batch size 1
    x1 = torch.randn(1, 4, 6)
    output1 = model(x1)
    assert output1.shape == x1.shape
    
    # Test with batch size 3
    x2 = torch.randn(3, 4, 6)
    output2 = model(x2)
    assert output2.shape == x2.shape
    
    # Test with larger dimensions
    model_large = BiN(num_features=20, seq_length=50)
    x3 = torch.randn(2, 20, 50)
    output3 = model_large(x3)
    assert output3.shape == x3.shape


def test_forward_pass_values():
    """Test that forward pass produces reasonable values"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    
    output = model(x)
    
    # Output should be finite
    assert torch.isfinite(output).all()
    
    # Output should not be all zeros
    assert not torch.allclose(output, torch.zeros_like(output))
    
    # Output should have reasonable scale (not too large/small)
    assert output.abs().max() < 100
    assert output.abs().min() < output.abs().max()


def test_temporal_normalization():
    """Test temporal normalization component"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    B, F, T = x.shape
    
    temporal_output = model._temporal_norm(x, B, F, T)
    
    # Should have same shape as input
    assert temporal_output.shape == x.shape
    
    # Should be finite
    assert torch.isfinite(temporal_output).all()


def test_feature_normalization():
    """Test feature normalization component"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    B, F, T = x.shape
    
    feature_output = model._feature_norm(x, B, F, T)
    
    # Should have same shape as input
    assert feature_output.shape == x.shape
    
    # Should be finite
    assert torch.isfinite(feature_output).all()


def test_scale_bias_application():
    """Test scale and bias application methods"""
    model = BiN(num_features=4, seq_length=6)
    normalized = torch.randn(2, 4, 6)
    B, F, T = normalized.shape
    
    # Test temporal scale/bias
    temporal_result = model._apply_temporal_scale_bias(normalized, B, F, T)
    assert temporal_result.shape == normalized.shape
    assert torch.isfinite(temporal_result).all()
    
    # Test feature scale/bias
    feature_result = model._apply_feature_scale_bias(normalized, B, F, T)
    assert feature_result.shape == normalized.shape
    assert torch.isfinite(feature_result).all()


def test_weight_constraints():
    """Test that mixing weights stay positive"""
    model = BiN(num_features=4, seq_length=6)
    
    # Manually set weights to negative
    model.feature_weight.data = torch.tensor([-0.5])
    model.temporal_weight.data = torch.tensor([-0.3])
    
    x = torch.randn(2, 4, 6)
    output = model(x)
    
    # Weights should be reset to positive values
    assert model.feature_weight.item() > 0
    assert model.temporal_weight.item() > 0
    assert torch.isfinite(output).all()


def test_zero_std_handling():
    """Test handling of zero standard deviation"""
    model = BiN(num_features=4, seq_length=6)
    
    # Create input with constant values along time (zero temporal std)
    x = torch.randn(2, 4, 1).repeat(1, 1, 6)  # Same value across time
    
    output = model(x)
    
    # Should not produce NaN or inf values
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()


def test_gradient_flow():
    """Test that gradients flow through the model"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6, requires_grad=True)
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    
    # Check that parameter gradients exist
    assert model.feature_bias.grad is not None
    assert model.feature_scale.grad is not None
    assert model.temporal_bias.grad is not None
    assert model.temporal_scale.grad is not None
    assert model.feature_weight.grad is not None
    assert model.temporal_weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_compatibility():
    """Test model on CUDA if available"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    
    # Test on CPU
    output_cpu = model(x)
    assert output_cpu.device.type == 'cpu'
    
    # Test on CUDA
    model_cuda = model.cuda()
    x_cuda = x.cuda()
    output_cuda = model_cuda(x_cuda)
    assert output_cuda.device.type == 'cuda'
    
    # Results should be similar (allowing for numerical differences)
    assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-6)


def test_cpu_behavior():
    """Test model behavior on CPU"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    
    output = model(x)
    assert output.device.type == 'cpu'
    assert torch.isfinite(output).all()


def test_edge_cases():
    """Test edge cases: single feature, single timestep, etc."""
    
    # Test with single feature
    model_single_feat = BiN(num_features=1, seq_length=6)
    x_single_feat = torch.randn(2, 1, 6)
    output_single_feat = model_single_feat(x_single_feat)
    assert output_single_feat.shape == x_single_feat.shape
    assert torch.isfinite(output_single_feat).all()
    
    # Test with single timestep
    model_single_time = BiN(num_features=4, seq_length=1)
    x_single_time = torch.randn(2, 4, 1)
    output_single_time = model_single_time(x_single_time)
    assert output_single_time.shape == x_single_time.shape
    assert torch.isfinite(output_single_time).all()
    
    # Test with large batch
    model_large_batch = BiN(num_features=4, seq_length=6)
    x_large_batch = torch.randn(100, 4, 6)
    output_large_batch = model_large_batch(x_large_batch)
    assert output_large_batch.shape == x_large_batch.shape
    assert torch.isfinite(output_large_batch).all()


def test_deterministic_behavior():
    """Test that model produces same output for same input"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(2, 4, 6)
    
    # Set model to eval mode to ensure deterministic behavior
    model.eval()
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
        
    # Should produce identical outputs
    assert torch.allclose(output1, output2)


def test_input_validation():
    """Test behavior with different input types and edge values"""
    model = BiN(num_features=4, seq_length=6)
    
    # Test with zeros
    x_zeros = torch.zeros(2, 4, 6)
    output_zeros = model(x_zeros)
    assert torch.isfinite(output_zeros).all()
    
    # Test with very large values
    x_large = torch.full((2, 4, 6), 1000.0)
    output_large = model(x_large)
    assert torch.isfinite(output_large).all()
    
    # Test with very small values
    x_small = torch.full((2, 4, 6), 1e-6)
    output_small = model(x_small)
    assert torch.isfinite(output_small).all()


# Pytest fixtures for common test data
@pytest.fixture
def standard_model():
    """Fixture providing a standard BiN model for testing"""
    return BiN(num_features=4, seq_length=6)


@pytest.fixture
def sample_input():
    """Fixture providing sample input data"""
    return torch.randn(2, 4, 6)


# Parametrized tests for different model sizes
@pytest.mark.parametrize("num_features,seq_length", [
    (1, 1),      # Minimal size
    (2, 3),      # Small size
    (10, 20),    # Medium size
    (40, 50),    # Bitcoin LOB size
])
def test_different_model_sizes(num_features, seq_length):
    """Test BiN with different model dimensions"""
    model = BiN(num_features=num_features, seq_length=seq_length)
    x = torch.randn(2, num_features, seq_length)
    
    output = model(x)
    
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 100])
def test_different_batch_sizes(batch_size):
    """Test BiN with different batch sizes"""
    model = BiN(num_features=4, seq_length=6)
    x = torch.randn(batch_size, 4, 6)
    
    output = model(x)
    
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


# Test using fixtures
def test_with_fixtures(standard_model, sample_input):
    """Test using pytest fixtures"""
    output = standard_model(sample_input)
    
    assert output.shape == sample_input.shape
    assert torch.isfinite(output).all() 