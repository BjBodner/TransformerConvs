import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDWConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1, padding=0, stride=1, nhead=4, embedding_dim=12, dim_feedforward=24) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.window_size = kernel_size ** 2
        self.channel_embedding_matrix = torch.nn.Embedding(in_channels, embedding_dim)
        self.positional_embedding_matrix = torch.nn.Embedding(kernel_size ** 2, embedding_dim)
        self.self_attn_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.linear = nn.Linear(embedding_dim, 1)
        self.fold_args = dict(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        self.unfold = nn.Unfold(**self.fold_args)


    def forward(self, x):

        output_size = x.size()[2:]

        # unfold into windows - (B, C, K**2, num_windows)
        out = self.unfold(x).reshape(x.size(0), self.in_channels, self.kernel_size ** 2, -1)

        # get channel and position indices
        window_size = out.size(1) // self.in_channels
        num_windows = out.size(-1)

        # note could be there is an issue here with the channel index, as it is not clear if the channel index should be the same for each window
        # or if it should be the same for each value in the window
        channel_idx = torch.arange(self.in_channels).reshape(1, -1, 1, 1)
        position_idx = torch.arange(self.window_size).reshape(1, 1, -1, 1)

        # get channel and position embeddings
        channel_embeddings = self.channel_embedding_matrix(channel_idx)
        position_embeddings = self.positional_embedding_matrix(position_idx)

        # TODO make value actual descrete value embeddings embeddings
        # using current solution of just copyting the original value across all embedding dimensions
        value_embeddings = out.unsqueeze(-1)

        # create window sequence - (B, C, K**2, num_windows, embedding_dim)
        windows_sequence = channel_embeddings + position_embeddings + value_embeddings

        # permute to (B, num_windows, C, K**2, embedding_dim)
        windows_sequence = windows_sequence.permute(0, 3, 1, 2, 4)

        # run through transformer layer
        orig_shape = windows_sequence.shape
        sequence_shape = (-1, windows_sequence.shape[3], windows_sequence.shape[4])
        out = self.self_attn_layer(windows_sequence.reshape(sequence_shape)).reshape(orig_shape)

        # aggregate over embedding dim - (B, num_windows, C, K**2)
        out = self.linear(out).squeeze()

        # fold windows and return to original shape
        out = out.permute(0, 2, 3, 1).reshape(x.size(0), self.in_channels * self.window_size, -1)
        out = F.fold(out, output_size=output_size, **self.fold_args)

        return out


class TransformerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, padding=0, stride=1, nhead=4, embedding_dim=12, dim_feedforward=24) -> None:
        super().__init__()
        self.depth_wise_conv = None
        if kernel_size > 1:
            self.depth_wise_conv = TransformerDWConv2d(in_channels, kernel_size, dilation, padding, stride, nhead, embedding_dim, dim_feedforward)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depth_wise_conv(x) if self.depth_wise_conv is not None else x
        out = self.point_wise_conv(out)
        return out

if __name__ == "__main__":
    a = torch.randn(1, 1, 3, 3)

    in_channels = 3
    out_channels = 32
    kernel_size = 3
    dilation = 1
    padding = 1
    stride = 1
    nhead=4
    embedding_dim = 4
    dim_feedforward = 8
    
    img_size = (16,16)
    batch_size = 2
    # input_tensor = torch.arange(batch_size * in_channels * img_size[0] * img_size[1]).reshape(batch_size, in_channels, img_size[0], img_size[1]).float()
    input_tensor = torch.randn(batch_size, in_channels, img_size[0], img_size[1])

    # point_wise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    standard_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

    tconv = TransformerConv2d(
        in_channels=in_channels, 
        out_channels=out_channels,
        kernel_size=kernel_size, 
        dilation=dilation, 
        padding=padding, 
        stride=stride, 
        nhead=nhead, 
        embedding_dim=embedding_dim, 
        dim_feedforward=dim_feedforward
    )

    output_tensor = tconv(input_tensor)
    a = 1

    num_params_tconv = sum(p.numel() for p in tconv.parameters() if p.requires_grad)
    num_params_standard_conv = sum(p.numel() for p in standard_conv.parameters() if p.requires_grad)

    a = 1