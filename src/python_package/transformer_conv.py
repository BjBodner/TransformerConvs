import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation=1, padding=0, stride=1, nhead=4, embedding_dim=12, dim_feedforward=24) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channel_embedding_matrix = torch.nn.Embedding(in_channels, embedding_dim)
        self.positional_embedding_matrix = torch.nn.Embedding(kernel_size ** 2, embedding_dim)
        self.self_attn_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.linear = nn.Linear(embedding_dim, 1)
        self.fold_args = dict(
            kernel_size = kernel_size,
            dilation = 1,
            padding = 0,
            stride = 1,
        )
        self.unfold = nn.Unfold(**self.fold_args)


    def forward(self, x):

        output_size = x.size()[2:]

        # unfold into windows
        out = self.unfold(x)

        # get channel and position indices
        num_windows = out.size(1) // self.in_channels
        window_size = out.size(2)
        channel_idx = torch.arange(self.in_channels).repeat_interleave(num_windows).reshape(1, -1, 1)
        position_idx = torch.arange(window_size).reshape(1, 1, -1)

        # get channel and position embeddings
        channel_embeddings = self.channel_embedding_matrix(channel_idx)
        position_embeddings = self.positional_embedding_matrix(position_idx)

        # TODO make value actual descrete value embeddings embeddings
        # using current solution of just copyting the original value across all embedding dimensions
        value_embeddings = out.unsqueeze(-1)

        # create window sequence
        windows_sequence = channel_embeddings + position_embeddings + value_embeddings

        # run through transformer layer
        orig_shape = windows_sequence.shape
        sequence_shape = (-1, windows_sequence.shape[2], windows_sequence.shape[3])
        out = self.self_attn_layer(windows_sequence.reshape(sequence_shape)).reshape(orig_shape)

        # aggregate over embedding dim
        out = self.linear(out).squeeze()

        # fold windows and return to original shape
        out = F.fold(out, output_size=output_size, **self.fold_args)

        return out

if __name__ == "__main__":
    a = torch.randn(1, 1, 3, 3)

    in_channels = 5
    kernel_size = 2
    nhead=4
    embedding_dim = 12
    dim_feedforward = 24
    input_tensor = torch.arange(2 * in_channels * 3 * 3).reshape(2, in_channels, 3, 3).float()


    tconv = TransformerConv2d(in_channels, kernel_size=kernel_size, nhead=nhead, embedding_dim=embedding_dim, dim_feedforward=dim_feedforward)

    output = tconv(input_tensor)
    a = 1


    # fold = nn.Fold(output_size=output_size, **fold_args)

    # # fold back into image
    # out = fold(out) 


    point_wise_conv_out = point_wise_conv(out)
    a = 1