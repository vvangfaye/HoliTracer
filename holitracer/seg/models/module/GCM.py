import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# pytorch 1.7.0
class Global_Context_Module(nn.Module):
    def __init__(self, channels, nHeads: int = 2, nLayers: int = 3):
        super(Global_Context_Module, self).__init__()
        assert len(channels) == 4  # 四组特征的通道数
        assert nLayers == 3
        c1, c2, c3, c4 = channels

        self.attention12 = nn.ModuleList()
        self.attention13 = nn.ModuleList()
        self.Squeue = nn.ModuleList()
        for c in (c2, c3, c4):
            self.attention12.append(nn.MultiheadAttention(embed_dim=c, num_heads=nHeads))
            self.attention13.append(nn.MultiheadAttention(embed_dim=c, num_heads=nHeads))
            self.Squeue.append(nn.Conv2d(c * nLayers, c, kernel_size=1, stride=1, padding=0))

    @staticmethod
    def att(q, kv, attention, saveName=None):
        bs, c, h, w = q.shape
        q = q.flatten(2).permute(2, 0, 1)  # h*w, bs, c
        kv = kv.flatten(2).permute(2, 0, 1)
        
        output, attn_weights = attention(q, kv, kv, need_weights=True)  # attn_weights: (bs, num_heads, h*w, h*w)
        output = output.permute(1, 2, 0).view(bs, c, h, w)
        
        attentions = None
        if saveName:
            nh = int(math.sqrt(attn_weights.shape[1]))
            # head_num = attn_weights.shape[1]
            # for i in range(head_num):
            #     attentions = attn_weights[0, i, :].reshape(nh, -1)
            #     attentions = F.interpolate(attentions.unsqueeze(0).unsqueeze(0),
            #                                size=(512, 512),
            #                                mode="nearest")[0][0]
            #     plt.imsave(fname=f"/home/faye/code/linerefactor/attention/{saveName}_head{i}.png", arr=attentions.detach().cpu().numpy(), format='png')
            # attentions = attn_weights[0, -1, :].reshape(nh, -1)
            # mean
            attentions = attn_weights.mean(dim=1)[0].reshape(nh, -1)
            # 可视化第一个 batch 和第一个头的 Attention 权重
            attentions = F.interpolate(attentions.unsqueeze(0).unsqueeze(0),
                                    size=(512, 512),
                                    mode="bicubic")[0][0]
            # plt.imsave(fname=f"/home/faye/code/linerefactor/attention/{saveName}.png", arr=attentions.detach().cpu().numpy(), format='png')
        
        return output, attentions
    
    def forward(self, f1, f2, f3, saveName=None):
        """
        :param f1: 来自小视场图像的特征，有四组 已经有梯度
        :param f2: 来自中视场图像的特征，有四组
        :param f3: 来自大视场图像的特征，有四组
        :return: 融合之后的特征，四组
        """
        outFeats = []
        attention_map1_sum = np.zeros((512, 512))
        attention_map2_sum = np.zeros((512, 512))
        for i, x in enumerate(zip(f1, f2, f3)):
            fx1, fx2, fx3 = x
            if i == 0:
                outFeats.append(fx1)
                continue
            fx2.requires_grad = True
            fx3.requires_grad = True
            if saveName:
                att12_saveName = saveName + '_2_' + str(i)
                att13_saveName = saveName + '_3_' + str(i)
            else:
                att12_saveName = None
                att13_saveName = None

            att1, attention_map1 = self.att(q=fx1, kv=fx2, attention=self.attention12[i - 1], saveName=att12_saveName)
            att2, attention_map2 = self.att(q=fx1, kv=fx3, attention=self.attention13[i - 1], saveName=att13_saveName)
            
            if saveName:
                attention_map1_sum += attention_map1.detach().cpu().numpy()
                attention_map2_sum += attention_map2.detach().cpu().numpy()
                
            Fx = torch.cat((fx1,
                            att1,
                            att2),
                           dim=1)
            outFeats.append(self.Squeue[i - 1](Fx))
            
        if saveName:
            plt.imsave(fname=f"./attention_map/{saveName}_2_map.png", arr=attention_map1_sum, format='png',cmap='jet')
            plt.imsave(fname=f"./attention_map/{saveName}_3_map.png", arr=attention_map2_sum, format='png',cmap='jet')
        return outFeats