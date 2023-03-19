import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, Function
# TODO - perhaps make this a double layer, to addapt to distributions, or use a gaussian prior
# TODO - Try reducing the number of parameters to make it look better


# def clamp(min_val, max_val):
#     class Clamp(Function):
#         @staticmethod
#         def forward(ctx, i):
#             ctx._mask = (i.ge(min_val) * i.le(max_val))
#             return i.clamp(min_val, max_val)

#         @staticmethod
#         def backward(ctx, grad_output):
#             mask = Variable(ctx._mask.type_as(grad_output.data))
#             return grad_output * mask

#     return Clamp().apply


class Clamp(Function):

    @staticmethod
    def forward(ctx, i, min, max):
        ctx._mask = (i.ge(min) * i.le(max))
        ctx.scale_dim = min.shape
        return i.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        if ctx.scale_dim == (1,):
            min_max_val_grad = 1 - mask.mean().reshape(ctx.scale_dim)
        else:
            min_max_val_grad = 1 - mask.mean((0, 2, 3)).reshape(ctx.scale_dim) # currently assumes 4d tensor
        # print(min_max_val_grad)
        return grad_output * mask, -min_max_val_grad, min_max_val_grad


def scale_clip(min_scale, max_scale):
    class _pq(Function):
        @staticmethod
        def forward(ctx, scale):
            ctx.save_for_backward(scale)
            return torch.clamp(scale, min=min_scale, max=max_scale)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()       
            scale, = ctx.saved_tensors

            penalty_grad = (scale - max_scale).clamp(min=0) + (scale - min_scale).clamp(max=0)
            # grad_scale = (grad_input * penalty_grad).sum()
            grad_scale = grad_input * penalty_grad
            # grad_scale = -grad_input 
            # grad_scale = penalty_grad
            # print(grad_scale)
            return grad_scale


    return _pq().apply

class RemapLayer(nn.Module):
    def __init__(self, num_embeddings, in_channels=None, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False):
        super().__init__()
        self.unsigned = unsigned
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_clipper = scale_clip(min_scale, max_scale)
        self.clamp = Clamp.apply

        if in_channels is None:
            self.idx_offset = 0
            self.num_embeddings = num_embeddings
            self.num_embeddings_per_channel = num_embeddings
            self.scale_dim = 1
        else:
            self.idx_offset = torch.arange(in_channels).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * num_embeddings
            self.num_embeddings_per_channel = num_embeddings
            self.num_embeddings = num_embeddings * in_channels
            self.scale_dim = in_channels

        self.value_embeddings = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=1)
        self.scale = torch.nn.Parameter(initial_scale * torch.ones(1, self.scale_dim, 1, 1))
        self.min_max_momentum = 0.9

    def update_min_max(self, x):
        # update min and max
        self.min_scale = self.min_scale * self.min_max_momentum + x.std() * (1 - self.min_max_momentum)
        self.max_scale = self.max_scale * self.min_max_momentum + x.max() * (1 - self.min_max_momentum)

    def forward(self, x):

        # clip scales
        self.update_min_max(x)
        scale = scale_clip(self.min_scale, self.max_scale)(self.scale)

        if self.unsigned:
            # map range to 0,1
            out_01 = x.clamp(min=torch.tensor(0), max=scale).div(scale)
        else:
            # map range to 0,1
            # out_01 = x.clamp(min=-scale, max=scale).div(scale).add(1).div(2)
            out_01 = self.clamp(x,-scale, scale).div(scale).add(1).div(2)
            
        # map range to 0, num_embeddings
        out3 = out_01 * (self.num_embeddings_per_channel - 1)

        # add offset for per-channel mapping
        out4 = out3 + self.idx_offset

        # get lower and upper indices of embedding values
        lower_1 = torch.floor(out4)
        upper_1 = torch.ceil(out4)


        # get embedding values
        lower_1_value = self.value_embeddings(lower_1.long()).squeeze(-1)
        upper_1_value = self.value_embeddings(upper_1.long()).squeeze(-1)


        # calculate differences
        diff_lower1 = out4 - lower_1
        diff_upper1 = 1 - diff_lower1

        # calculate interpolation
        interp_value = diff_lower1 * lower_1_value + diff_upper1 * upper_1_value

        return interp_value


if __name__ == "__main__":

    num_embeddings = 64
    embedding_dim = 1
    offset = 0.0
    scale = 3.0
    
    batch_size = 3
    num_channels = 4
    img_size = (32,32)
    # input_tensor = torch.relu(torch.randn(batch_size, num_channels, img_size[0], img_size[1]))
    input_tensor = 2.5 * torch.randn(batch_size, num_channels, img_size[0], img_size[1])

    model = RemapLayer(num_embeddings=num_embeddings, in_channels=num_channels, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False)
    
    lr = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_true = 0.5 * input_tensor ** 3 + 0.1 * input_tensor ** 2 + 3 * input_tensor + 1
    y_true = y_true + (torch.randn_like(y_true)**2) * 0.1

    for i in range(1000):
        optimizer.zero_grad()
        y_pred = model(input_tensor)

        loss = criterion(y_true, y_pred)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"loss: {loss.item()}, scale: {model.scale.mean()}")


    out = model(input_tensor)



    learned_mappings = [model.value_embeddings.weight.data[i * num_embeddings : (i + 1) * num_embeddings, 0].numpy() for i in range(num_channels)] 
    learned_scales = model.scale.data.squeeze().tolist()
    learned_x = [np.linspace(-scale, scale, num_embeddings) for scale in learned_scales]

    # plot learned mappings
    fig, ax = plt.subplots(1, num_channels, figsize=(num_channels * 5, 5))
    for i in range(num_channels):
        ax[i].plot(learned_x[i], learned_mappings[i])
        ax[i].set_title(f"channel {i}")
        ax[i].set_xlabel("input")
        ax[i].set_ylabel("output")
    plt.show()
    a = 1
