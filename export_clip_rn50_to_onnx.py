import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class MyAttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = flatten_start_dim_2(x).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.view(x.shape[1:])


def make_avgpool2d_from_conv(n_channels, kernel_size):
    avgpool2d = nn.Conv2d(
        n_channels, n_channels, kernel_size, stride=kernel_size, bias=False
    )

    avgpool2d.weight.data = torch.zeros_like(avgpool2d.weight.data)
    for i in range(avgpool2d.weight.shape[0]):
        avgpool2d.weight.data[i, i] = (
            torch.ones_like(avgpool2d.weight.data[i, i]) / kernel_size**2
        )

    avgpool2d.weight.data = avgpool2d.weight.data.detach()
    return avgpool2d


def flatten_start_dim_2(x):
    size_product = x.size(2) * x.size(3)
    x = x.view(x.size(0), x.size(1), size_product)
    return x


class MyVisionModel(nn.Module):
    def __init__(self, model):
        super(MyVisionModel, self).__init__()
        self.model = model
        self.avgpool2d_64_2 = make_avgpool2d_from_conv(64, 2)

        for i, (name, layer) in enumerate(self.model.named_modules()):
            if isinstance(layer, clip.model.Bottleneck) and isinstance(
                layer.avgpool, nn.AvgPool2d
            ):
                layer.avgpool = make_avgpool2d_from_conv(
                    layer.conv2.out_channels, layer.avgpool.kernel_size
                )
                if layer.downsample:
                    if layer.downsample[0].kernel_size == 1:
                        layer.downsample[0] = nn.Identity()
                    else:
                        layer.downsample[0] = make_avgpool2d_from_conv(
                            layer.conv1.in_channels, layer.downsample[0].kernel_size
                        )
        self.model.layer1[0].downsample[0] = nn.Identity()

        self.attnpool = MyAttentionPool2d(1, 1, 1, 1)

        self.attnpool.positional_embedding = self.model.attnpool.positional_embedding
        self.attnpool.k_proj = self.model.attnpool.k_proj
        self.attnpool.q_proj = self.model.attnpool.q_proj
        self.attnpool.v_proj = self.model.attnpool.v_proj
        self.attnpool.c_proj = self.model.attnpool.c_proj
        self.attnpool.num_heads = self.model.attnpool.num_heads

    def forward(self, x):
        def stem(x):
            x = self.model.relu1(self.model.bn1(self.model.conv1(x)))
            x = self.model.relu2(self.model.bn2(self.model.conv2(x)))
            x = self.model.relu3(self.model.bn3(self.model.conv3(x)))
            x = self.avgpool2d_64_2(x)
            return x

        x = stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.attnpool(x)
        return x


if __name__ == "__main__":
    filepath = "clip_rn50.onnx"
    input_tensor = torch.randn((1, 3, 224, 224))

    m, preprocess = clip.load("RN50")

    model = m.visual.float().eval()

    m = MyVisionModel(model).eval()

    torch.onnx.export(
        m.eval(),
        input_tensor,
        filepath,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
