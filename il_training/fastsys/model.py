import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from einops.layers.torch import Rearrange


class SelfAttention(nn.Module):
    def __init__(self, input_dim=180, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, ray_features):
        # ray_features: (B, 180) → (B, Seq=1, Feat=180)
        ray_features = ray_features.unsqueeze(1)
        attn_out, _ = self.mha(ray_features, ray_features, ray_features)
        return self.norm(ray_features + attn_out).squeeze(1)

class ChannelWiseSelfAttention(nn.Module):
    def __init__(self, channel_dim, num_heads=4):
        """
        :param channel_dim: 输入特征通道数（即embed_dim）
        :param num_heads: 注意力头数
        """
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=num_heads,
            batch_first=False  # 使用 (seq_len, batch, features) 格式
        )
        self.norm = nn.LayerNorm(channel_dim)

    def forward(self, x):
        """
        输入形状: (batch, channels, seq_len)
        输出形状: (batch, channels, seq_len)
        """
        # 调整维度为 (seq_len, batch, channels)
        x_perm = x.permute(2, 0, 1)
        
        # 自注意力计算
        attn_output, _ = self.mha(x_perm, x_perm, x_perm)
        
        # 残差连接 + 层归一化
        output = self.norm(x_perm + attn_output)
        
        # 恢复维度 (batch, channels, seq_len)
        return output.permute(1, 2, 0)

class ChannelWise2DSelfAttention(nn.Module):
    def __init__(self, channel_dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channel_dim,
            num_heads=num_heads,
            batch_first=False
        )
        self.norm = nn.LayerNorm(channel_dim)

    def forward(self, x):
        """处理4D卷积特征输入"""
        B, C, H, W = x.shape
        
        # 将空间维度展平为序列 [B, C, H, W] → [B, C, L] (L=H*W)
        x_flat = x.view(B, C, -1)  # 形状变为 (B, C, L)
        
        # 调整维度为 (L, B, C) → 适配MultiheadAttention输入
        x_perm = x_flat.permute(2, 0, 1)
        
        # 自注意力计算
        attn_output, _ = self.mha(x_perm, x_perm, x_perm)
        
        # 残差连接 + 归一化
        output = self.norm(x_perm + attn_output)
        
        # 恢复形状 [L, B, C] → [B, C, H, W]
        output = output.permute(1, 2, 0).view(B, C, H, W)
        return output

class CrossAttention(nn.Module):
    def __init__(self, query_dim=128, key_dim=256, num_heads=4):
        super().__init__()
        # 将状态特征投影到与 Occ 相同的维度（可选）
        self.state_proj = nn.Linear(query_dim, key_dim)
        
        # 多头交叉注意力（状态作为 Query，Occ 作为 Key/Value）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=key_dim,
            num_heads=num_heads,
            kdim=key_dim,
            vdim=key_dim
        )
        
        # 层归一化与残差连接
        self.norm = nn.LayerNorm(key_dim)
        
    def forward(self, query_feat, key_feat):
        """
        Args:
            query_feat: 状态特征 (B, state_dim)
            occ_feat: Occ 特征 (B, occ_dim)
        Returns:
            attn_occ: 状态引导后的 Occ 特征 (B, occ_dim)
        """
        # 维度调整：PyTorch 的 MultiheadAttention 需要 (Seq, B, Dim)
        query = self.state_proj(query_feat).unsqueeze(0)  # (1, B, occ_dim)
        key = value = key_feat.unsqueeze(0)               # (1, B, occ_dim)
        
        # 计算交叉注意力
        attn_output, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False
        )
        
        # 残差连接 + 层归一化
        attn_output = self.norm(key_feat + attn_output.squeeze(0))
        return attn_output

class TripleCrossAttention(nn.Module):
    def __init__(self, state_dim=128, occ_dim=256, ray_dim=128):
        super().__init__()
        # L1: State → Occ
        self.state2occ = CrossAttention(query_dim=state_dim, key_dim=occ_dim)
        # L2: State → Raycast
        self.state2ray = CrossAttention(query_dim=state_dim, key_dim=ray_dim)
        # L3: Occ → Raycast
        self.occ2ray = CrossAttention(query_dim=occ_dim, key_dim=ray_dim)
        # 融合层
        self.fuse = nn.Linear(occ_dim + ray_dim + state_dim, 512)
        
    def forward(self, occ_feat, state_feat, ray_feat):
        # L1: 状态条件化的 Occ 特征
        occ_attn = self.state2occ(state_feat, occ_feat)  # (B, 256)
        # L2: 状态条件化的 Raycast 特征
        ray_attn = self.state2ray(state_feat, ray_feat)  # (B, 128)
        # L3: Occ 引导的 Raycast 增强
        ray_occ = self.occ2ray(occ_attn, ray_attn)       # (B, 128)
        # 拼接所有特征
        combined = torch.cat([occ_attn, ray_occ, state_feat], dim=1)
        return self.fuse(combined)

class PolicyNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.min_action = torch.tensor(cfg['data']['min_action']).cuda()
        self.max_action = torch.tensor(cfg['data']['max_action']).cuda()
        
        class ResNet18Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                # 加载预训练的resnet50结构，但不加载权重
                self.backbone = resnet18(weights=None)
                # 修改第一层卷积以适配单通道输入
                self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                # 获取最后fc层输入特征数
                self.feature_dim = self.backbone.fc.in_features
                # 去掉最后的fc层和avgpool，只保留到flatten前
                self.backbone = nn.Sequential(
                    self.backbone.conv1,
                    self.backbone.bn1,
                    self.backbone.relu,
                    self.backbone.maxpool,
                    self.backbone.layer1,
                    self.backbone.layer2,
                    self.backbone.layer3,
                    self.backbone.layer4,
                    # ChannelWise2DSelfAttention(512),  # ResNet18 layer4 输出通道为512
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                # 后接一个线性层，输出256维特征
                self.fc = nn.Sequential(
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU()
                )

            def forward(self, x):
                x = self.backbone(x)
                x = self.fc(x)
                return x

        self.obs_encoder = ResNet18Backbone()

        
        # self.ray_cast_encoder = nn.Sequential(
        #     nn.Linear(180, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        # )
        
        self.ray_cast_encoder = nn.Sequential(
            # 添加通道维度 [B, 180] -> [B, 1, 180]
            Rearrange("b l -> b 1 l"),
            
            # 三层 1D 卷积
            nn.LazyConv1d(out_channels=64, kernel_size=5, padding=1, padding_mode='circular'), 
            nn.ELU(),
            nn.LazyConv1d(out_channels=128, kernel_size=5, stride=2, padding=1, padding_mode='circular'), 
            nn.ELU(),
            nn.LazyConv1d(out_channels=256, kernel_size=5, stride=2, padding=1, padding_mode='circular'),
            nn.ELU(),
            
            # ChannelWiseSelfAttention(256),
            
            # 展平 + 全连接
            Rearrange("b c l -> b (c l)"),
            nn.LazyLinear(256),
        )
        
        
        # 状态处理分支 (vx,vy,vyaw,x,y,yaw + target_x,target_y)
        self.state_processor = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # 融合决策层
        self.decision_net = nn.Sequential(
            # nn.Linear(256 + 128, 128),
            # nn.Linear(512, 128),
            nn.Linear(256 + 256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # 输出在[0,1]范围
        )
    
    def forward(self, state, target, obstacle, ray_cast, ray_cast_occ):

        # 处理障碍物地图
        # img_feat = self.obs_encoder(obstacle)  # 增加通道维度
        
        # 处理状态和目标
        state_target = torch.cat([state, target], dim=1)
        state_feat = self.state_processor(state_target)
        ray_cast_feat = self.ray_cast_encoder(ray_cast)
        ray_cast_occ_feat = self.obs_encoder(ray_cast_occ)
        
        # 特征融合
        # combined = torch.cat([img_feat, state_feat], dim=1)
        combined = torch.cat([ray_cast_occ_feat, state_feat, ray_cast_feat], dim=1)
        # combined = torch.cat([state_feat, ray_cast_feat], dim=1)
        action = self.decision_net(combined)

        return action
    
    def denormalize(self, action_norm):
        return action_norm * (self.max_action - self.min_action) + self.min_action