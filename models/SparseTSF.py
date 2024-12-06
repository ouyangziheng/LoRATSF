import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.rank = 2
        assert self.model_type in ["linear", "mlp"]

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.k = int(self.seg_num_x * self.seg_num_y * 0.9)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        if self.model_type == "linear":
            self.linearA = nn.Linear(self.seg_num_x, self.rank, bias=False)
            self.linearB = nn.Linear(self.rank, self.seg_num_y, bias=False)

        elif self.model_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        # 获取 linearA 的权重
        weight_A = self.linearA.weight

        # 获取 linearB 的权重
        weight_B = self.linearB.weight
        weight = weight_B @ weight_A

        topk_values, topk_indices = torch.topk(weight.flatten(), self.k)

        modified_weight = torch.zeros_like(weight)
        modified_weight.flatten()[topk_indices] = topk_values

        self.linearC = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        self.linearC.weight = torch.nn.Parameter(modified_weight)

        # 1D convolution aggregation
        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == "linear":
            y = self.linearC(x)
        elif self.model_type == "mlp":
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
