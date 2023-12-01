import torch
import numpy as np
import os
import random
from graphviz import Digraph
from torch.autograd import Variable, Function


def get_device():
    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    return device

def get_optimizer(optimizer, params, lrs):
    if len(lrs)==0: return None
    opt = { 'none': None, 'sgd' : torch.optim.SGD, 'adam' : torch.optim.Adam }[optimizer]
    if opt is None: return opt
    opt = opt([ {'params':[list(params.values())[i]], 'lr':lrs[i] } for i in range(len(params)) ])
    return opt

def get_scheduler(scheduler, optimizer, epochs):
    if optimizer is None:
        return None
    sch = { 'none': None, 'cosine': torch.optim.lr_scheduler.CosineAnnealingLR }[scheduler]
    if sch is None: return sch
    sch = sch( optimizer, T_max=epochs+1)
    return sch

def get_criterion(criterion, threshold=None):
    c = None
    if criterion == 'l1':
        c = torch.nn.L1Loss()
    elif criterion == 'l2':
        c = torch.nn.L1Loss()
    elif criterion == 'huber':
        c = torch.nn.HuberLoss(delta=threshold)
    elif criterion == 'l1sat':
        c = SaturatedL1(threshold=threshold)
    elif criterion == 'l2sat':
        c = SaturatedL2(threshold=threshold)
    return c



def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

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

# used for generate graph detecting none gradients
# get_dot = register_hooks(loss)
# loss.backward()
# dot = get_dot()
# dot.save('tmp.dot') # to get .dot
# dot.render('tmp') # to get SVG
def register_hooks(var):

    def iter_graph(root, callback):
        queue = [root]
        seen = set()
        while queue:
            fn = queue.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    queue.append(next_fn)
            callback(fn)

    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot


# CUSTOM LOSS
class SaturatedL1(torch.nn.Module):
    def __init__(self, threshold):
        super(SaturatedL1, self).__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        absolute_errors = torch.abs(predictions - targets)
        # absolute_errors = torch.norm(predictions - targets, p=1, dim=-1)
        absoulte_errors = torch.clamp(absolute_errors, max=self.threshold)
        return absoulte_errors.mean()

class SaturatedL2(torch.nn.Module):
    def __init__(self, threshold):
        super(SaturatedL2, self).__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        absolute_errors = torch.pow(predictions - targets, 2)
        # absolute_errors = torch.norm(predictions - targets, p=2, dim=-1)
        absoulte_errors = torch.clamp(absolute_errors, max=self.threshold)
        return absoulte_errors.mean()
