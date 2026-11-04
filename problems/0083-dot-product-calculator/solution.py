import torch

def calculate_dot_product(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (torch.Tensor): 1D tensor representing the first vector.
        vec2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        torch.Tensor: The dot product of the two vectors as a scalar tensor.
    """
    vector_a = vec1
    vector_b = vec2
    dot_product = torch.dot(vector_a, vector_b)
    return dot_product