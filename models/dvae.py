import torch.nn as nn
import torch
import torch.nn.functional as F

# from knn_cuda import KNN 这个没有用到

# from pointnet2_ops import pointnet2_utils
# from pointnet2.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

# from pointnet2_ops import pointnet2_utils
# from pointnet2
from Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

from .build import MODELS
from utils import misc
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
# from extensions.emd import emd
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *



from knn_cuda import KNN
knn = KNN(k=4, transpose_mode=False)


class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1) 

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 256),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   ) # B C N 分组数---32 256 32 4，卷积核为1，输出大小和输入大小相同

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 1024),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                nn.GroupNorm(4, output_channel),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

    # todo 由32 3 32和32 128 32 --》 32 256 32 4 坐标是绝对坐标，但特征全都是相对坐标得到的特征
    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        """
        Args:
            coor_q: 32 3 32 # 全局特征的中心点
            x_q: 32 128 32 # 全局特征的点云特征
            coor_k: 32 3 32 # 局部特征的中心点
            x_k: 32 128 32 # 局部特征的点云特征

            首先利用 32 3 32 自己查自己k个近邻点，得到索引idx，然后idx加上索引，索引范围等于（N*B），，KNN索引的范围是0~B，所以要加上乘N的偏移
            然后对x_k转置，32 32 128，目的是方便BN相乘
            再x_k根据idx索引查找特征，得到的特征是32*32*4 128，得到【相对位置的特征】feature 4096 128-- 32 128 32 4

            接着x_q作为【没有knn的x_k，原始相对坐标特征】登场，32 128 32 1，扩展为32 128 32 4，广播到每个近邻点
            feature - x_q



        Returns:

        """

        # coor: bs, 3, np, x: bs, c, np

        k = 4
        batch_size = x_k.size(0) # 32
        num_points_k = x_k.size(2) # 32 中心点的个数
        num_points_q = x_q.size(2) # 32 全局点数量

        with torch.no_grad():
            # todo : 索引值范围由点数确定，32，范围0~31
            _, idx = knn(coor_k, coor_q)  # bs k np 为q里面的点查找k的中的近邻点
            # 输入：32 3 32, 32 3 32 返回的idx: 32 4 32
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k # 即 32 1 1 给每个中心点加上的偏移量
            idx = idx + idx_base # todo 每个样本加上【0，32，64，，992】的偏移，因为要求查找x_k的近邻点特征
            idx = idx.view(-1) # 32*32*4=4096
        num_dims = x_k.size(1) # 特征向量的维度是128
        x_k = x_k.transpose(2, 1).contiguous() # 32 32 128
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :] # 32*32 128 索引大小为[4096,]，允许重复选取同一行，【1024，128】-》【4096，128】
        """
            idx是在（32，3，32）里面提取到的k=4的近邻点索引，这个索引为什么能拿来查找特征x_k呢？
            32 3 32-32 32 3---- 32 32 128，这里的两个32都是代表相同的批大小，以及fps查找的32个中心点
            32 32 3 代表同一批数据的坐标
            32 32 128 代表同一批数据的特征
            因此由32 32 3得来的 32 4 32的索引可以用来查找32 32 128的特征，
            特征变换为(1024,128)后，索引展平为(4096,)
            这个索引展平为(4096,)后，可以用来查找特征的每一行
        """
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous() # 32，4，32，128-》32 128 32 4
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k) # 32 128 32 4 扩充最后一个维度为原来的4倍，广播中心点
        # TODO x_k是相对坐标的特征，，然后融合了knn的4个近邻点，
        # TODO x_q也是相对坐标的特征，没有融合4个近邻点，所以相对于x_k来说类似于全局特征
        feature = torch.cat((feature - x_q, x_q), dim=1) # 融合中心点绝对坐标和相对坐标的局部特征
        return feature # 32 256 32 4 即 32 256 32 k

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3 

        # bs 3 N   bs C N
        feature_list  = []
        coor = coor.transpose(1, 2).contiguous()         # B 3 N 即 32 32 3-》 32 3 32
        f = f.transpose(1, 2).contiguous()               # B C N 即 32 32 1024-》 32 1024 32
        f = self.input_trans(f)             # B 128 N 即 32 1024 32 -> 32 128 32 一维卷积

        # todo 生成【微观局部坐标】和【宏观局部坐标】融合的特征，实际上就是【带4个邻近点的点K和原本的点K进行特征融合】
        f = self.get_graph_feature(coor, f, coor, f) # B 256 N k 处理点云特征并提取局部-全局关系 32 256 32 4
        # todo 输入 32 128 32 以及 32 3 32
        # todo 输出 32 256 32 4
        f = self.layer1(f)                           # B 256 N k 即 32 256 32 4
        f = f.max(dim=-1, keepdim=False)[0]          # B 256 N 即 聚合最后一维 32 256 32 1 == 32 256 32
        feature_list.append(f) # feature_list只是f的容器，只有长度

        f = self.get_graph_feature(coor, f, coor, f) # B 512 N k 得到 32 512
        # todo 输入 32 256 32 以及 32 3 32
        # todo 输出 32 512 32 4
        f = self.layer2(f)                           # B 512 N k 也是 32 512 32 4
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N 聚合最后一维 32 512 32 1 == 32 512 32
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        # todo 输入 32 512 32 以及 32 3 32
        # todo 输出 32 1024 32 4
        f = self.layer3(f)                           # B 512 N k # 32 512 32 4
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N # 聚合最后一维 32 512 32 1 == 32 512 32
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        # todo 输入 32 512 32 以及 32 3 32
        # todo 输出 32 1024 32 4
        f = self.layer4(f)                           # B 1024 N k # 32 1024 32 4
        f = f.max(dim=-1, keepdim=False)[0]          # B 1024 N # 聚合最后一维 32 1024 32 1 == 32 1024 32
        feature_list.append(f)

        f = torch.cat(feature_list, dim = 1)         # B 2304 N
        # todo 32 2304 32 【256+512+512+1024】

        f = self.layer5(f)                           # B C' N # 即 32 8192 32 # num_tokens: 8192,
        
        f = f.transpose(-1, -2) # 32 32 8192

        return f

### ref https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py ###
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    

"""
    将输入点云(B×N×3)划分为多个局部点云组(B×G×M×3)
    通过FPS(最远点采样)选择G个中心点,使用KNN为每个中心点找到K个最近邻
    中心化处理，使每个组的中心点为原点
"""
class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3 即 32 32 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M 即 32 32 8
        assert idx.size(1) == self.num_group # 32个组
        assert idx.size(2) == self.group_size # 每组8个点
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :] # 32*32 3索引[32*32*8,]，允许重复选取同一行
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2) # 32 32 8 3
        return neighborhood, center

"""
    将局部点云组编码为特征向量
"""
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        # TODO：以下都是相对坐标，拼接的也是相对坐标的全局特征和局部特征
        bs, g, n , _ = point_groups.shape # 32 32 8
        point_groups = point_groups.reshape(bs * g, n, 3) # 32*32 8 3
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n 即 32*32 3 8 -》  32*32 256 8
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1 即 32*32 256 1 聚合附件8个点的特征
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        ### 即 32*32 256 8 （-1表示维度保持不变）【全局特征】和 32*32 256 8【局部位置】连接，让局部位置获得全局信息
        # todo 这里的全局和局部都是依据相对位置坐标来的，无非是自己聚合，再和自己连接
        feature = self.second_conv(feature) # BG 1024 n 即 32*32 1024 8
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel) # 32 32 1024

"""
    将编码后的特征向量解码为粗糙点云和精细点云
    从特征向量重建点云
    生成粗粒度点云(coarse)
    使用折叠操作(folding)生成细粒度点云(fine)
    基于2×2网格种子点和特征向量重建局部点云
"""
class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine # 32
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4 # 32/4=8
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024), # encoder_channel=256
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2) # 1 2 S


    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3
        
        '''
        bs, g, c = feature_global.shape # 32 32 256
        feature_global = feature_global.reshape(bs * g, c) #  32*32 256

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3) # BG M 3 即 1024 8 3
        # mlp-1024,24,1-》1024,8,3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3 即 1024 8 4 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N # 即 1024 3 32

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1) # BG 2 M (S) 1024 2 8
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N 1024 2 32

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine) # BG 1024 N=1024 256 32
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # BG C N
        ## 1024 256+2+3 32
    
        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1) # BG (M) S 3= 1024 8 4 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1) # BG 3 N = 1024 3 32

        fine = self.final_conv(feat) + center   # BG 3 N = 1024 3 32
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2) # 32 32 32 3
        coarse = coarse.reshape(bs, g, self.num_coarse, 3) # 32 32 8 3
        return coarse, fine

"""
    离散变分自编码器(Discrete VAE)
    codebook：离散token的嵌入表
    loss_func_cdl1/cdl2：Chamfer距离损失函数
"""
@MODELS.register_module()
class DiscreteVAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel = self.encoder_dims, output_channel = self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims)) # 8192 256

        self.dgcnn_2 = DGCNN(encoder_channel = self.tokens_dims, output_channel = self.decoder_dims)
        self.decoder = Decoder(encoder_channel = self.decoder_dims, num_fine = self.group_size) # num_fine=32 encoder_channel=256
        self.build_loss_func()

        
        
    def build_loss_func(self):
        self.loss_func_cdl1 = ChamferDistanceL1().cuda()
        self.loss_func_cdl2 = ChamferDistanceL2().cuda()
        # self.loss_func_emd = emd().cuda()

    def recon_loss(self, ret, gt):
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        coarse = coarse.reshape(bs*g, -1, 3).contiguous()
        fine = fine.reshape(bs*g, -1, 3).contiguous()
        group_gt = group_gt.reshape(bs*g, -1, 3).contiguous()

        loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
        loss_fine_block = self.loss_func_cdl1(fine, group_gt)

        loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):

        # reconstruction loss
        loss_recon = self.recon_loss(ret, gt)
        # kl divergence
        logits = ret[-1] # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device = gt.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)), None, None, 'batchmean', log_target = True)

        return loss_recon, loss_klv


    def forward(self, inp, temperature = 1., hard = False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)   #  B G C 32 32 1024
        # todo 聚合微小【宏观局部坐标特征】与【微观局部坐标特征】，得到的还是局部坐标的特征
        logits = self.dgcnn_1(logits, center) #  B G N 即 32 32 8192，此处N是码本大小
        soft_one_hot = F.gumbel_softmax(logits, tau = temperature, dim = 2, hard = hard) # B G N # 大小是 32 32 8192的独热向量
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook) # B G C (32,32,8192)->(8192,256)==》32 32 256
        feature = self.dgcnn_2(sampled, center) # 32 32 256
        coarse, fine = self.decoder(feature)
        # fine =  32 32 32 3
        # coarse = 32 32 8 3


        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3) # 32 1024 3
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3) # 32 256 3

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret

