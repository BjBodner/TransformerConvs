import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable, Function
# TODO - perhaps make this a double layer, to addapt to distributions, or use a gaussian prior


class Clamp(Function):

    @staticmethod
    def forward(ctx, i, scale):
        # ctx._mask = (i.ge(min.detach()) * i.le(max.detach()))
        ctx._mask = i.div(scale).abs().le(1)
        ctx.scale_dim = scale.shape
        ctx.input = i
        # This is where the extra source of the scale gradient comes from, need to do this with an STE
        # so that the gradient flows normally to the input, and doesn't affect the scales
        return i.clamp(-scale.detach(), scale.detach()) 

    @staticmethod
    def backward(ctx, grad_output):
        mask = Variable(ctx._mask.type_as(grad_output.data))
        sign = torch.sign(ctx.input)

        grad_scale = (1 - mask) * grad_output * sign
        # grad_scale = 1 - mask  
        if ctx.scale_dim[1] == 1:
            min_max_val_grad = grad_scale.mean().reshape(ctx.scale_dim)
        else:
            min_max_val_grad = grad_scale.mean((0, 2, 3)).reshape(ctx.scale_dim) # currently assumes 4d tensor
        # print(min_max_val_grad)
        min_max_val_grad = min_max_val_grad
        return grad_output, min_max_val_grad


def scale_clip(min_scale, max_scale, rel_eps=1e-3):
    class _pq(Function):
        @staticmethod
        def forward(ctx, scale):
            eps = rel_eps * (max_scale - min_scale)

            ctx.save_for_backward(scale)
            return torch.clamp(scale, min=min_scale * (1 + eps), max=max_scale * (1 - eps))

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()       
            scale, = ctx.saved_tensors

            # penalty_grad = (scale - max_scale).clamp(min=0) + (scale - min_scale).clamp(max=0)
            
            
            
            penalty_grad = -1 / (scale - max_scale) - 1 / (scale - min_scale)
            # penalty_grad = -0.5 * penalty_grad
            # grad_scale = (grad_input * penalty_grad).sum()
            grad_scale =  0.5 * penalty_grad + grad_input

            eps = rel_eps * (max_scale - min_scale)
            scale.data = torch.clamp(scale.data, min=min_scale * (1 + eps), max=max_scale * (1 - eps))
            # grad_scale = -grad_input 
            # grad_scale = penalty_grad
            # print(grad_scale)
            # grad_scale = 0.000001 * grad_scale
            # print(scale)
            return grad_scale


    return _pq().apply



def clip_01(min_scale, max_scale, rel_eps=1e-1):
    class _clip01(Function):
        @staticmethod
        def forward(ctx, i, scale):
            eps = rel_eps * (max_scale - min_scale)

            # assert torch.all(scale < max_scale) and torch.all(scale > min_scale)
            scale = torch.clamp(scale, min=min_scale * (1 + eps), max=max_scale * (1 - eps))

            i_minus_1_plus_1 = i.div(scale)
            ctx._mask = i_minus_1_plus_1.abs().le(1)
            ctx.scale_dim = scale.shape
            ctx.input = i
            ctx.scale = scale

            i_minus_1_plus_1_ = (i_minus_1_plus_1.clamp(-1,1) - i_minus_1_plus_1).detach() + i_minus_1_plus_1

            return i_minus_1_plus_1_.add(1).div(2)  

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            mask = Variable(ctx._mask.type_as(grad_output.data))
            sign = torch.sign(ctx.input)
            scale = ctx.scale

            # grad_scale = (1 - mask) * grad_output * sign
            # grad_scale = 1 - mask  
            if ctx.scale_dim[1] == 1:
                min_max_val_grad = -(1 - mask).mean().reshape(ctx.scale_dim)
            else:
                min_max_val_grad = -(1 - mask).mean((0, 2, 3)).reshape(ctx.scale_dim) # currently assumes 4d tensor

            penalty_grad = -1 / (scale - max_scale) - 1 / (scale - min_scale)

            grad_scale =  0.5 * penalty_grad + min_max_val_grad
            # grad_scale = torch.clamp(grad_scale, min=-0.1, max=0.1)


            assert torch.all(scale < max_scale) and torch.all(scale > min_scale)

            activations_decay = 0.001 * ctx.input
            return grad_input * mask + activations_decay, grad_scale
        
    return _clip01().apply

def gradient_scale(x, scale=0.1):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    return y


class RemapLayer(nn.Module):
    def __init__(self, num_embeddings, in_channels=None, min_scale=2.5, max_scale=3.5, initial_scale=3., unsigned=False):
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
        self.min_max_momentum = 0.999

    def update_min_max(self, x):
        # update min and max
        self.min_scale = self.min_scale * self.min_max_momentum + 2 * x.std().detach() * (1 - self.min_max_momentum)
        self.max_scale = self.max_scale * self.min_max_momentum + x.abs().max().detach() * (1 - self.min_max_momentum)

    def gradient_scale(self, x, scale=0.1):
        yout = x
        ygrad = x * scale
        y = (yout - ygrad).detach() + ygrad
        return y

    def forward(self, x):

        # clip scales
        self.update_min_max(x)

        # scale = self.gradient_scale(self.scale, scale=0.01)
        # scale = scale_clip(self.min_scale, self.max_scale)(self.scale)
        # self.scale.data = scale.data
        # scale = self.scale
        # print(scale, self.scale)
        if self.unsigned:
            # map range to 0,1
            out_01 = x.clamp(min=torch.tensor(0), max=scale).div(scale)
        else:
            # map range to 0,1
            # out_01 = x.clamp(min=-scale, max=scale).div(scale.detach()).add(1).div(2)
            # out_01 = self.clamp(x, scale).div(scale).add(1).div(2)
            # out_01 = self.clamp(x, scale).div(scale).add(1).div(2)
            

            out_01 = clip_01(self.min_scale, self.max_scale)(x, self.scale)

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
    
    input_scale = 1.
    
    for experiment in range(10):
        print("\n\nExperiment: ", experiment)
        # model = RemapLayer(num_embeddings=num_embeddings, in_channels=None, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False)
        
        model = nn.Sequential(
                # RemapLayer(num_embeddings=num_embeddings),
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(num_channels),
                # RemapLayer(num_embeddings=num_embeddings),
                nn.ReLU(),
                # nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, padding=0),
                # nn.BatchNorm2d(num_channels),
                # RemapLayer(num_embeddings=num_embeddings),

        )
        

        lr = 0.01

        model_params = []
        for name, params in model.named_parameters():
            if 'scale' in name:
                model_params += [{'params': [params], 'lr': 1.0 * lr, 'weight_decay': 0}]
            else:
                model_params += [{'params': [params], 'weight_decay': 0.0}]

        optimizer = torch.optim.Adam(model_params, lr=lr)
        criterion = nn.MSELoss()


        for i in range(1000):

            input_tensor = input_scale * torch.randn(batch_size, num_channels, img_size[0], img_size[1])
            y_true = 0.5 * input_tensor ** 3 + 0.1 * input_tensor ** 2 + 3 * input_tensor + 1
            y_true = y_true + (torch.randn_like(y_true)**2) * 0.1


            optimizer.zero_grad()
            y_pred = model(input_tensor)

            loss = criterion(y_true, y_pred)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                try:
                    print(f"loss: {loss.item()}, scale: {model[2].scale.mean().item(), model[2].min_scale.mean().item(), model[2].max_scale.mean().item()}")
                except AttributeError:
                    print(f"loss: {loss.item()}")


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
