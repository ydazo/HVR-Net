import torch
import torch.nn as nn
import torch.nn.functional as F


class cross_Sim(nn.Module): 
    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s**3
              
    def forward(self, Fx, Fy, wins=None):
        if wins:
            self.wins = wins
            self.win_len = wins**3
        b, c, d, h, w = Fy.shape
      
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors), -1).type(torch.FloatTensor)
   
        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(Fx.device)

        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2  # 1

        Fx = F.pad(Fx,  tuple_(pd, length=6)) 
     
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        attn = (Fy @ Fx.transpose(-2, -1))
        sim = attn.softmax(dim=-1)
        out = (sim @ G) 
        out = rearrange(out , 'b (d h w) 1 c -> b c d h w', d=d,h=h,w=w)
    
        return out

class VCIM(nn.Module):
    """Channel attention implemented with a lightweight 3D self-attention
    that computes channel descriptors by attending along the channel subspace.

    This follows the reference pattern but extended to 3D volumes:
    - Optionally apply a small spatial downsample (sr_ratio) to reduce cost.
    - Compute q/k/v via 1x1x1 convs, reshape to (B, head_num, head_dim, N)
    - Compute attn = q @ k^T (shape (B, head_num, head_dim, head_dim)), softmax,
      then apply to v to get (B, head_num, head_dim, N).
    - Reshape back to (B, C, D, H, W), global-average over spatial to get (B,C,1,1,1)
    - Pass through a small ca_gate to produce final channel weights.
    """

    def __init__(self, in_channels, head_num=4, reduction=4, sr_ratio=1):
        super(VCIM, self).__init__()
        self.in_channels = in_channels
        self.head_num = head_num
        # ensure head_dim is integer
        assert in_channels % head_num == 0, "in_channels must be divisible by head_num"
        self.head_dim = in_channels // head_num
        self.sr_ratio = sr_ratio

        # optional spatial reduction (use avg pool to reduce spatial tokens)
        if sr_ratio > 1:
            self.downsample = nn.AvgPool3d(kernel_size=sr_ratio, stride=sr_ratio)
        else:
            self.downsample = nn.Identity()

        # q/k/v generators
        self.norm = nn.BatchNorm3d(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)

        # small gating MLP producing final channel weights
        self.ca_gate = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.attn_drop = nn.Identity()
        self.scaler = self.head_dim ** -0.5

    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape

        y = self.downsample(x)  # maybe reduce spatial tokens
        B, C, Dp, Hp, Wp = y.shape
        N = Dp * Hp * Wp

        y = self.norm(y)

        q = self.q(y).view(B, self.head_num, self.head_dim, N)  # (B, head_num, head_dim, N)
        k = self.k(y).view(B, self.head_num, self.head_dim, N)
        v = self.v(y).view(B, self.head_num, self.head_dim, N)

        # attn: (B, head_num, head_dim, head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaler
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # (B, head_num, head_dim, N)
        out = torch.matmul(attn, v)

        # reshape back to (B, C, Dp, Hp, Wp)
        out = out.view(B, self.head_num * self.head_dim, Dp, Hp, Wp)

        # upsample back to original spatial size if downsampled
        if self.sr_ratio > 1:
            out = F.interpolate(out, size=(d, h, w), mode='trilinear', align_corners=False)

        # global pooling to get channel descriptor (B, C, 1, 1, 1)
        ch_desc = out.mean(dim=(2, 3, 4), keepdim=True)

        # generate channel weights and apply
        weights = self.ca_gate(ch_desc)  # (B, C, 1, 1, 1)
        return x * weights

class CASM(nn.Module):
    def __init__(self, in_channels=64):
        super(CASM, self).__init__()
        # 支持奇偶通道数：将通道分为 c1 / c2
        c1 = in_channels // 2
        c2 = in_channels - c1
        # 为两部分分别建立 conv2d（3x3 和 5x5），以支持通道为奇数时的分割
        self.conv2dk3_a = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.conv2dk3_b = nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False)
        self.conv2dk5_a = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.conv2dk5_b = nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._c1 = c1
        self._c2 = c2
        
    def forward(self, x):
        # 三个方向平均池化 -> 投影
        # x_avg: (B, C, H, W)  D 方向投影
        # y_avg: (B, C, D, W)  H 方向投影（将 H 视作“高度”）
        # z_avg: (B, C, D, H)  W 方向投影
        x_avg = torch.mean(x, dim=2, keepdim=False)  # (B,C,H,W)
        y_avg = torch.mean(x, dim=3, keepdim=False)  # (B,C,D,W)
        z_avg = torch.mean(x, dim=4, keepdim=False)  # (B,C,D,H)

        # 按预先在 __init__ 中计算的 c1/c2 分割通道，使用对应的 conv
        c1, c2 = self._c1, self._c2
        x_a, x_b = torch.split(x_avg, [c1, c2], dim=1)
        y_a, y_b = torch.split(y_avg, [c1, c2], dim=1)
        z_a, z_b = torch.split(z_avg, [c1, c2], dim=1)

        # 对每部分分别应用 3x3 / 5x5
        x_out_a = self.conv2dk3_a(x_a)
        x_out_b = self.conv2dk5_b(x_b)
        y_out_a = self.conv2dk3_a(y_a)
        y_out_b = self.conv2dk5_b(y_b)
        z_out_a = self.conv2dk3_a(z_a)
        z_out_b = self.conv2dk5_b(z_b)

        # 恢复通道维度
        x_out = torch.cat([x_out_a, x_out_b], dim=1)  # (B,C,H,W)
        y_out = torch.cat([y_out_a, y_out_b], dim=1)  # (B,C,D,W)
        z_out = torch.cat([z_out_a, z_out_b], dim=1)  # (B,C,D,H)

        # 扩回对应维度以便广播到 (B,C,D,H,W)
        x_att = self.sigmoid(x_out).unsqueeze(2)  # (B,C,1,H,W)
        y_att = self.sigmoid(y_out).unsqueeze(3)  # (B,C,D,1,W)
        z_att = self.sigmoid(z_out).unsqueeze(4)  # (B,C,D,H,1)

        # 按元素相乘，得到最终的空间注意力 (B,C,D,H,W)
        att = x_att * y_att * z_att

        # 返回与输入按元素相乘的结果
        return x * att + x    
class VFRB(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7, head_num=4,sr_ratio=2):
        super(VFRB, self).__init__()
        self.dcov = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
        self.channel_attention = VCIM(in_channels, reduction)
        self.spatial_attention = CASM(in_channels)
        
    def forward(self, x):
        x1 = self.dcov(x)
        x2 = self.conv(x)
        
        x = x1+x2+self.bn(x)
        
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x




# 测试模块
if __name__ == "__main__":
    input_tensor = torch.randn(8, 64, 16, 32, 32)  # (batch_size, channels, depth, height, width)

    cbam1 = VFRB(64)
    output1 = cbam1(input_tensor)
    print("Output1 shape:", output1.shape)