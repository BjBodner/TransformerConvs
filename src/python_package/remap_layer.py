import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

def scale_clip(min_scale, max_scale):
    class _pq(torch.autograd.Function):
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
            grad_scale = penalty_grad
            return grad_scale


    return _pq().apply

class RemapLayer(nn.Module):
    def __init__(self, num_embeddings, in_channels=None, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False):
        super().__init__()
        self.unsigned = unsigned
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_clipper = scale_clip(min_scale, max_scale)

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


    def forward(self, x):

        # clip scales
        scale = self.scale_clipper(self.scale)

        if self.unsigned:
            # map range to 0,1
            out_01 = x.clamp(min=torch.tensor(0), max=scale).div(scale)
        else:
            # normalize inputs
            # out = (x - x.mean()) / x.std()

            # map range to 0,1
            out_01 = x.clamp(min=-scale, max=scale).div(scale).add(1).div(2)
            
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

        # if self.embedding_dim == 1:

        # else:
            

        #     lower_1_value = self.value_embeddings(lower_1.long())
        #     upper_1_value = self.value_embeddings(upper_1.long())   
        # lower_1_value = self.value_embeddings(lower_1.long()).squeeze(-1)
        # upper_1_value = self.value_embeddings(upper_1.long()).squeeze(-1)

        # calculate differences
        diff_lower1 = out4 - lower_1
        diff_upper1 = 1 - diff_lower1

        # calculate interpolation
        interp_value = diff_lower1 * lower_1_value + diff_upper1 * upper_1_value

        return interp_value


if __name__ == "__main__":

    num_embeddings = 256
    embedding_dim = 1
    offset = 0.0
    scale = 3.0
    
    batch_size = 2
    num_channels = 4
    img_size = (32,32)
    # input_tensor = torch.relu(torch.randn(batch_size, num_channels, img_size[0], img_size[1]))
    input_tensor = torch.randn(batch_size, num_channels, img_size[0], img_size[1])

    model = RemapLayer(num_embeddings=num_embeddings, in_channels=num_channels, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False)
    
    lr = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_true = 5 * input_tensor ** 2 + 3 * input_tensor + 1
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
