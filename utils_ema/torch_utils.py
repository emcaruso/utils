import torch

def repeat_tensor_to_match_shape( source_tensor, target_shape ):
    
    # Calculate the number of dimensions to add
    num_dims_to_add = len(target_shape) - len(source_tensor.shape)

    # Add dimensions to the source tensor
    for _ in range(num_dims_to_add):
        source_tensor = source_tensor.unsqueeze(0)

    # Expand the source tensor to get the target shape
    repeated_tensor = source_tensor.expand(*target_shape)

    return repeated_tensor
