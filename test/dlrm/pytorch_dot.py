import torch
import numpy as np

batch_size = 64
num_rows = 27
num_cols = 128

def export(tensor, name):
    t = tensor.clone().detach().cpu().numpy()
    print("{} shape {}, isnan {}".format(name, t.shape, (t != t).any()))
    t.tofile(name)

def random_uniform(shape):
    return torch.from_numpy(np.random.uniform(low=-1, high=1, size=shape)).to(dtype).to('cuda')

dtype = torch.float32
name_suffix = "fp32" if dtype == torch.float else "fp16"

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

input = random_uniform((batch_size, num_rows, num_cols))

export(input, "data/input_" + name_suffix)

input.requires_grad = True

mlp_input = input.narrow(1, 0, 1).squeeze()
tinput = torch.transpose(input, 1, 2)

interaction = torch.bmm(input, tinput)
tril_indices_row, tril_indices_col = torch.tril_indices(
        interaction.shape[1], interaction.shape[2], offset=-1)
interaction_flat = interaction[:, tril_indices_row, tril_indices_col]
interaction_output = torch.cat([mlp_input, interaction_flat], dim=1)

export(interaction_output, "data/output_" + name_suffix)

mlp_input_grad = None
def mlp_input_hook(g):
    global mlp_input_grad
    mlp_input_grad = g.clone().detach()

mlp_input.register_hook(mlp_input_hook)

grad = random_uniform(interaction_output.shape)
interaction_output.backward(grad)

mlp_input_grad_pad = mlp_input_grad

zero_tensor = torch.zeros((num_rows - 1) * num_cols, dtype=dtype, device='cuda')
pad_tensors = []
for b in range(batch_size):
    pad = torch.cat([mlp_input_grad_pad[b], zero_tensor])
    pad_tensors.append(pad)

mlp_input_grad_pad = torch.cat(pad_tensors).view(batch_size, num_rows, num_cols)
input_grad = input.grad.sub(mlp_input_grad_pad)

export(grad, "data/input_grad_" + name_suffix)
export(input_grad, "data/output_input_grad_" + name_suffix)
export(mlp_input_grad, "data/output_mlp_input_grad_" + name_suffix)
