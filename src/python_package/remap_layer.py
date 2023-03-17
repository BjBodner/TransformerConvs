import torch.nn as nn
import torch

# TODO - add support for per-channel mapping
# TODO stabalize with running mean and var of scale

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
    def __init__(self, num_embeddings, embedding_dim=1, min_scale=2.5, max_scale=3.5, initial_scale=5., unsigned=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.value_embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.unsigned = unsigned
        # self.scale = torch.nn.Parameter(torch.tensor((max_scale + min_scale) / 2))
        self.scale = torch.nn.Parameter(torch.tensor(initial_scale))
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_clipper = scale_clip(min_scale, max_scale)

    def forward(self, x):

        # clip scales
        scale = self.scale_clipper(self.scale)


        if self.unsigned:
            # map range to 0,1
            out_01 = x.clamp(min=torch.tensor(0), max=scale).div(scale)
        else:
            # normalize inputs
            out = (x - x.mean()) / x.std()

            # map range to 0,1
            out_01 = out.clamp(min=-scale, max=scale).div(scale).add(1).div(2)
            
        # map range to 0, num_embeddings
        out3 = out_01 * (self.num_embeddings - 1)

        # get lower and upper indices of embedding values
        lower_1 = torch.floor(out3)
        upper_1 = torch.ceil(out3)

        # get embedding values
        lower_1_value = self.value_embeddings(lower_1.long()).squeeze(-1)
        upper_1_value = self.value_embeddings(upper_1.long()).squeeze(-1)

        # calculate differences
        diff_lower1 = out3 - lower_1
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
    num_channels = 2
    img_size = (16,16)
    # input_tensor = torch.relu(torch.randn(batch_size, num_channels, img_size[0], img_size[1]))
    input_tensor = torch.randn(batch_size, num_channels, img_size[0], img_size[1])

    model = RemapLayer(num_embeddings=num_embeddings, unsigned=False)
    
    lr = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    y_true = 5 * input_tensor ** 3 + 3 * input_tensor + 1
    y_true = y_true + (torch.randn_like(y_true)**2) * 0.1

    for i in range(1000):
        optimizer.zero_grad()
        y_pred = model(input_tensor)

        loss = criterion(y_true, y_pred)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"loss: {loss.item()}, scale: {model.scale.item()}")


    out = model(input_tensor)

    a = 1
