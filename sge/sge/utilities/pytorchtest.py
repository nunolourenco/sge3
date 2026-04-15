import torch
import math
import numpy as np

_min_domain = -1
_max_domain = 1
_domain_delta = _max_domain - _min_domain
dtype = torch.float32

if torch.cuda.is_available():
    cur_dev = torch.device("cuda")
elif torch.backends.mps.is_available():
    cur_dev = torch.device("mps")
else:
    cur_dev = torch.device("cpu")

res = [8, 8]

def node_tensor(x1, dims=[]):
    return torch.tensor(x1, dtype=dtype, device=cur_dev)

def node_var(dimensions, n):
    temp = np.ones(len(dimensions), dtype=int)
    temp[n] = -1
    res = torch.reshape(torch.arange(0, dimensions[n], dtype=dtype), tuple(temp))
    resolution = dimensions[n]
    dimensions[n] = 1
    res = torch.add(torch.full(res.shape, _min_domain), res, alpha=((1.0 / (resolution - 1)) * _domain_delta))
    return res.repeat(tuple(dimensions))

def node_abs(x1, dims=[]):
    return torch.abs(x1)

def node_add(x1, x2, dims=[]):
    return torch.add(x1, x2)

def node_sub(x1, x2, dims=[]):
    return torch.sub(x1, x2)

def node_mul(x1, x2, dims=[]):
    return torch.mul(x1, x2)

def node_div(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.div(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def node_bit_and(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_and(left_child, right_child), 1e-6)

def node_bit_or(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_or(left_child, right_child), 1e-6)

def node_bit_xor(x1, x2, dims=[]):
    left_child = torch.mul(x1, 1e6).int()
    right_child = torch.mul(x2, 1e6).int()
    return torch.mul(torch.bitwise_xor(left_child, right_child), 1e-6)

def node_cos(x1, dims=[]):
    return torch.cos(torch.mul(x1, math.pi))

def node_sin(x1, dims=[]):
    return torch.sin(torch.mul(x1, math.pi))

def node_tan(x1, dims=[]):
    return torch.tan(torch.mul(x1, math.pi))

def node_exp(x1, dims=[]):
    return torch.exp(x1)

def node_if(x1, x2, x3, dims=[]):
    print(x3)
    return torch.where(x3 < 0, x1, x2)

def node_log(x1, dims=[]):
    return torch.where(x1 > 0.0, torch.log(x1), torch.tensor(-1.0, dtype=dtype, device=cur_dev))

def node_max(x1, x2, dims=[]):
    return torch.max(x1, x2)

def node_min(x1, x2, dims=[]):
    return torch.min(x1, x2)

def node_mdist(x1, x2, dims=[]):
    return torch.mul(torch.add(x1, x2), 0.5)

def node_mod(x1, x2, dims=[]):
    return torch.where(x2 != 0.0, torch.fmod(x1, x2), torch.tensor(0.0, dtype=dtype, device=cur_dev))
    #return torch.fmod(x1, x2)

def node_neg(x1, dims=[]):
    return torch.neg(x1)

def node_pow(x1, x2, dims=[]):
    return torch.where(x1 != 0, torch.pow(torch.abs(x1), torch.abs(x2)), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def node_inv(x1, dims=[]):
    return torch.where(x1 != 0, torch.div(torch.tensor(1.0, dtype=dtype, device=cur_dev), x1), torch.tensor(1.0, dtype=dtype, device=cur_dev))

def node_sign(x1, dims=[]):
    return torch.sign(x1)

def node_sqrt(child1, dims=[]):
    return torch.where(child1 > 0, torch.sqrt(child1), torch.tensor(0.0, dtype=dtype, device=cur_dev))

def tensor_rmse(x1, x2):
    x1 = torch.mul(x1, 1/127.5)
    x2 = torch.mul(x2, 1/127.5)
    return torch.sqrt(torch.mean(torch.square(torch.sub(x1, x2))))

def node_clamp(tensor):
    return torch.clip(tensor, _min_domain, _max_domain)

def node_sstepp(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)

    return torch.add(torch.mul(x2,
                              torch.sub(torch.mul(torch.mul(x, x2), 6.0 * _domain_delta),
                                        torch.sub(torch.mul(x2, 15.0 * _domain_delta),
                                                  torch.mul(x, 10.0 * _domain_delta)))),
                     _min_domain)

def node_sstep(x1, dims=[]):
    x = node_clamp(x1)
    x2 = torch.square(x)
    return torch.add(torch.sub(torch.mul(3.0 * _domain_delta, x2),
                               torch.mul(2.0 * _domain_delta, torch.mul(x2, x))),
                  torch.tensor(_min_domain, dtype=dtype, device=cur_dev))

def node_step(x1, dims=[]):
    return torch.where(x1 < 0.0,
                    torch.tensor(-1.0, dtype=dtype, device=cur_dev),
                    torch.tensor(1.0, dtype=dtype, device=cur_dev))

def node_frac(x1, dims=[]):
    return torch.frac(x1)

def node_len(x1, x2, dims=[]):
    return torch.hypot(x1, x2)

def node_lerp(x1, x2, x3, dims=[]):
    return torch.lerp(x1, x2, node_frac(node_abs(x3)))

def node_stack(nums, dimensions, edims):
    return torch.stack([torch.full(dimensions[:edims], float(carvar), dtype=dtype, device=cur_dev) for carvar in nums], dim = edims)

def final_transform_domain1(final_tensor):
    final_tensor = torch.where(torch.isnan(final_tensor), torch.tensor(0.0, dtype=dtype, device=cur_dev), final_tensor)

    final_tensor = node_clamp(final_tensor)
    final_tensor = torch.sub(final_tensor, _min_domain)
    final_tensor = torch.mul(final_tensor, 255 / _domain_delta)

    return final_tensor

var_x = node_var(np.copy(res), 0)
var_y = node_var(np.copy(res), 1)

torch_tests = []
tf_tests = []
torch_vars = []
tf_vars = []

torch_vars.append(torch.tensor(np.array([[-1, 2.2, 3], [0, 5.5, -6.9]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[1, 2, 3], [4, 5, 0]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[1, 1, 0], [1, 0, 1]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[0.3, 0.5, 1.2], [0.2, 0.7, 0.6]]), dtype=dtype, device=cur_dev))
torch_vars.append(torch.tensor(np.array([[-2, 3], [4.5, float('-inf')], [0, float('nan')]]), dtype=dtype, device=cur_dev))

cnt = 0
