import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, group_num:int = 16, eps:float = 1e-10):
        super(GroupBatchnorm2d,self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num

        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        # torch.broadcast_tensors()
        N, C, H,W = x.size()  # 获取输入张量的尺寸
        x = x.reshape(N, self.group_num, -1)   # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.reshape(N, C, H, W)  # 恢复原始形状

        return x * self.gamma + self.beta  # 返回归一化后的张量
    
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.depthwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
    def forward(self, x):
        out=self.depthwise_conv(x)
        out=self.pointwise_conv(out)
        return out
    


class SRM(nn.Module):
    def __init__(self,
                 oup_channels: int,  # 输出通道数
                 gate_treshold: float = 0.5,  # 门控阈值，默认为0.5
                 ):
        super().__init__()  # 调用父类构造函数

        # 使用BatchNorm
        self.bn = nn.BatchNorm2d(oup_channels)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigmoid = nn.Sigmoid()  # 创建 sigmoid 激活函数

    def forward(self, x):
        # 应用批量归一化
        bn_x = self.bn(x)

        # 重量权重计算
        reweights = self.sigmoid(bn_x)  # 在这里，可能需要根据实际情况调整计算方式

        info_mask = reweights >= self.gate_treshold  # W1(有信息量)
        noninfo_mask = reweights < self.gate_treshold  # W2（非信息量）
        x_1 = info_mask * x
        x_2 = noninfo_mask * x

        x = self.reconstruct(x_1, x_2)  # 重构特征
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # N,C/2,H,W
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # N,C/2,H,W
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 重构特征并连接


class CRM(nn.Module):
    def __init__(self, op_channel:int, alpha:float = 1/2, squeeze_radio:int = 2, group_size:int = 2, group_kernel_size:int = 3):
        super().__init__()

        self.up_channel   = up_channel = int(alpha * op_channel)
        self.low_channel  = low_channel = op_channel - up_channel
        self.squeeze1    = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2    = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # 上层特征转换
        self.GWC      = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1, padding=group_kernel_size // 2, groups=group_size)
        self.PWC1      = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        self.DSC1 = DepthwiseSeparableConvolution(up_channel // squeeze_radio,op_channel)
        # 下层特征转换
        self.PWC2      = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.DSC2 = DepthwiseSeparableConvolution(low_channel // squeeze_radio,low_channel // squeeze_radio)
        self.advavg     = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        # 分割输入特征,1*1卷积降维
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 上层特征转换
        Y1 = self.GWC(up) + self.DSC1(up)
        # 下层特征转换
        Y2 = torch.cat([self.PWC2(low), self.DSC2(low)], dim=1)

        # 特征融合
        out = torch.cat([Y1, Y2], dim=1)
        attention = F.softmax(self.advavg(out), dim=1)
        out = attention * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SFAM(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.sru = SRM(inp)
        self.cru = CRM(inp)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_sigmoid()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.sru(self.pool_h(x))
        x_w = self.cru(self.pool_w(x).permute(0, 1, 3, 2))

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
