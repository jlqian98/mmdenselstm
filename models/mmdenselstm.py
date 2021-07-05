import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=128, C=12, n_layer=2):
        super(LSTM, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.inconv = nn.Conv1d(C, 1, kernel_size=1)        # feature map -> 1
        self.deconv = nn.Conv1d(1, C, kernel_size=1)        # feature map -> C
        self.linear = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        """
        args:
            x: [M, C, N, T]
        return:
            out: [M, C, N, T]
        """
        M, C, N, T = x.shape

        x = x.view(M, C, N*T)           
        x = self.inconv(x)  # [M, 1, N*T]
        x = x.view(M, N, T).transpose(1, 2) # (M, T, N)
        out, (h_n, c_n) = self.lstm(x)  # [M, T, H]
        out = self.linear(out)  # [M, T, N]
        emb = out[:, -1, :] # [M, N]
        out = out.transpose(1, 2)   
        out = out.contiguous().view(M, 1, N*T)
        out = self.deconv(out)  # [M, C, N*T]
        out = out.view(M, C, N, T)
        
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm2', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(num_input_features, growth_rate, kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        prev_features = torch.cat(prev_features, 1)
        new_features = self.conv2(self.relu2(self.norm2(prev_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.num_input_features = num_input_features
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers['denselayer%d' % (i + 1)] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return features[-1]

class MMDenseLSTM(nn.Module):
    def __init__(self, input_channel=1,drop_rate=0.1, F=512):
        super(MMDenseLSTM, self).__init__()
        kl_low = [(14,4), (16,4), (16,4), (16,4), (16,4), (16,4), (16, 4)]
        kl_high = [(10,3), (10,3), (10,3), (10,3), (10,3), (10,3), (16, 3)]
        # kl_high = [(14,4), (14,4), (14,4), (14,4), (14,4), (14,4), (14, 4)]
        kl_full = [(6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2), (6, 2),]
        Hl, Hh, Hf = 32, 32, 64
        self.lowNet = _MMDenseLSTM_STEM(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3, kl=kl_low, drop_rate=drop_rate, H=Hl)
        self.highNet = _MMDenseLSTM_STEM(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3, kl=kl_high, drop_rate=drop_rate, H=Hh)
        self.fullNet = _MMDenseLSTM_STEM(input_channel=input_channel, first_channel=32, first_kernel=(3, 3), scale=3, kl=kl_full, drop_rate=drop_rate, H=Hf)
        last_channel = self.lowNet.channels[-1] + self.fullNet.channels[-1]
        self.out = nn.Sequential(
            _DenseBlock(
                2, last_channel, 32, drop_rate),
            nn.Conv2d(32, 4, 1)
        )


    def forward(self, input):

        B, C, F, T = input.shape
        low_input = input[:, :, :F // 2, :]
        high_input = input[:, :, F // 2:, :]
        low = self.lowNet(low_input)
        high = self.highNet(high_input)
        output = torch.cat([low, high ], 2)
        full_output = self.fullNet(input)
        output = torch.cat([output, full_output], 1)
        output = self.out(output)
        return output


class _MMDenseLSTM_STEM(nn.Module):
    def __init__(self, input_channel=1,
                 first_channel=32,
                 first_kernel=(3, 3),
                 scale=3,
                 kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                 drop_rate=0.1,
                 H=32):
        super(_MMDenseLSTM_STEM, self).__init__()
        self.first_channel = 32
        self.first_kernel = first_kernel
        self.scale = scale
        self.kl = kl
        self.first_conv = nn.Conv2d(input_channel, first_channel, 3, padding=1)
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.dense_padding = nn.ModuleList()
        self.dense_layers = nn.ModuleList()
        self.channels = [self.first_channel]
        ## [_,d1,...,ds,ds+1,u1,...,us]
        for k, l in kl[:scale + 1]:
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1], k, drop_rate))
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(k, k, kernel_size=(1, 1)),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            )
            self.channels.append(k)
        for i, (k, l) in enumerate(kl[scale + 1:]):
            self.upsample_layers.append(
                nn.ConvTranspose2d(self.channels[-1], self.channels[-1], kernel_size=2, stride=2))
            self.dense_layers.append(_DenseBlock(
                l, self.channels[-1]+self.channels[-(2+2*i)], k, drop_rate))
            self.channels.append(k)

        self.lstm = LSTM(in_dim=H, C=kl[len(kl)//2][0])
        # self.out = nn.Sequential(
        #     _DenseBlock(
        #         2, self.channels[-1], out_growth_rate, drop_rate),
        #     nn.Conv2d(out_growth_rate, input_channel, 1)
        # )

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        ## stem
        output = self.first_conv(input)
        dense_outputs = []

        ## downsample way
        for i in range(self.scale):
            output = self.dense_layers[i](output)
            dense_outputs.append(output)
            output = self.downsample_layers[i](output)  ## downsample

        ## upsample way
        output = self.dense_layers[self.scale](output)

        output = self.lstm(output)

        for i in range(self.scale):
            output = self.upsample_layers[i](output)
            output = self._pad(output, dense_outputs[-(i + 1)])
            output = torch.cat((output, dense_outputs[-(i + 1)]),dim=1)
            output = self.dense_layers[self.scale + 1 + i](output)
        # output = self._pad(output, input)
        return output
