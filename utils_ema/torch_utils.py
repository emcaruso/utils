import torch
import numpy as np

def repeat_tensor_to_match_shape( source_tensor, target_shape ):
    nparr = isinstance(source_tensor, np.ndarray)
    
    # Calculate the number of dimensions to add
    num_dims_to_add = len(target_shape) - len(source_tensor.shape)

    # Add dimensions to the source tensor
    for _ in range(num_dims_to_add):
        source_tensor = source_tensor.unsqueeze(0)

    # Expand the source tensor to get the target shape
    repeated_tensor = source_tensor.expand(*target_shape)

    if nparr:
        repeated_tensor = repeated_tensor.numpy()

    return repeated_tensor

def collate_fn(batch_list):
    # get list of dictionaries and returns input, ground_truth as dictionary for all batch instances
    batch_list = zip(*batch_list)
    all_parsed = []
    for entry in batch_list:
        if type(entry[0]) is dict:
            # make them all into a new dict
            ret = {}
            for k in entry[0].keys():
                ret[k] = torch.stack([obj[k] for obj in entry])
            all_parsed.append(ret)
        else:
            all_parsed.append(torch.LongTensor(entry))
    return tuple(all_parsed)

def dict_to_device( dict, device ):
    for k,v in dict.items():
        dict[k] = v.to(device)
    return dict

def print_cuda_mem_info():

    # t = torch.cuda.get_device_properties(0).total_memory
    # r = torch.cuda.memory_reserved(0)
    # a = torch.cuda.memory_allocated(0)
    # f = r-a  # free inside reserved
    # print( "total memory: ", t*0.001, " KiB")
    # print( "reserved memory: ", r*0.001, " KiB")
    # print( "allocated memory: ", a*0.001, " KiB")
    # print( "free inside reserved memory: ", f*0.001, " KiB")

    print(torch.cuda.memory_summary())

