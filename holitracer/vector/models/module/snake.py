import torch.nn as nn
import torch


class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()

        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj * 2 + 1)

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [input[..., -self.n_adj :], input, input[..., : self.n_adj]], dim=2
            )
        return self.fc(input)


class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(
            state_dim,
            out_state_dim,
            kernel_size=self.n_adj * 2 + 1,
            dilation=self.dilation,
        )

    def forward(self, input):
        if self.n_adj != 0:
            input = torch.cat(
                [
                    input[..., -self.n_adj * self.dilation :],
                    input,
                    input[..., : self.n_adj * self.dilation],
                ],
                dim=2,
            )
        return self.fc(input)


_conv_factory = {"grid": CircConv, "dgrid": DilatedCircConv}


class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()
        if conv_type == "grid":
            self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj)
        else:
            self.conv = _conv_factory[conv_type](
                state_dim, out_state_dim, n_adj, dilation
            )
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type="dgrid"):
        super(Snake, self).__init__()
        self.head = BasicBlock(feature_dim, state_dim, conv_type)
        self.res_layer_num = 7
        dilation = [1, 1, 1, 2, 2, 4, 4]
        n_adj = 4
        for i in range(self.res_layer_num):
            if dilation[i] == 0:
                conv_type = "grid"
            else:
                conv_type = "dgrid"
            conv = BasicBlock(
                state_dim, state_dim, conv_type, n_adj=n_adj, dilation=dilation[i]
            )
            self.__setattr__("res" + str(i), conv)

        fusion_state_dim = 256

        self.fusion = nn.Conv1d(
            state_dim * (self.res_layer_num + 1), fusion_state_dim, 1
        )

        self.prediction = nn.Sequential(
            nn.Conv1d(state_dim * (self.res_layer_num + 1) + fusion_state_dim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1),
        )

    def forward(self, x):
        # x: (B, NUM_FEATURES, NUM_POINTS)
        states = []
        x = self.head(x)
        states.append(x)
        for i in range(self.res_layer_num):
            x = self.__getattr__("res" + str(i))(x) + x
            states.append(x)

        state = torch.cat(states, dim=1)

        global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
        global_state = global_state.expand(
            global_state.size(0), global_state.size(1), state.size(2)
        )
        state = torch.cat([global_state, state], dim=1)

        x = self.prediction(state)

        return x
