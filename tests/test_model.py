import torch
import pytest
from model.network import MNISTNet
from train import train_model
import torch.nn.utils.prune as prune

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    model = MNISTNet()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Check output shape
    assert output.shape == (1, 10), "Output shape should be (1, 10)"
    
    # Check parameter count
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_training():
    accuracy, _ = train_model()
    assert accuracy >= 95.0, f"Model accuracy {accuracy:.2f}% is less than required 95%"

if __name__ == '__main__':
    pytest.main([__file__]) 