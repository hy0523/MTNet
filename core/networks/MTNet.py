import math
import sys

sys.path.append('../../')
sys.path.append('../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.networks.encoder import build_encoder
from core.networks.decoder.fpn import FPN
import copy
import time
from core.networks.util_modules.conv_block import ConvBNReLU


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(nn.Module):
    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class WindowMultiHeadedAttention(nn.Module):
    def __init__(self, tokensize, d_model, head, p=0.1, zone=2):
        super(WindowMultiHeadedAttention, self).__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize
        self.zone = zone

    def forward(self, x, t):
        bt, n, c = x.size()
        b = bt // t
        c_h = c // self.head
        query = self.query_embedding(x)  # [bt, n, c]
        key = self.key_embedding(x)
        value = self.value_embedding(x)
        key = key.view(b, t, self.zone, self.h // self.zone, self.zone, self.w // self.zone, self.head,
                       c_h)  # [b,t,s,h/s,s,w/s,h,per]
        key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, self.zone ** 2, self.head, -1,
                                                                    c_h)  # [b,s*s,h,t*h/s*w/s,per]
        query = query.view(b, t, self.zone, self.h // self.zone, self.zone, self.w // self.zone, self.head, c_h)
        query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, self.zone ** 2, self.head, -1, c_h)
        value = value.view(b, t, self.zone, self.h // self.zone, self.zone, self.w // self.zone, self.head, c_h)
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, self.zone ** 2, self.head, -1, c_h)
        att, _ = self.attention(query, key, value)
        att = att.view(b, self.zone, self.zone, self.head, t, self.h // self.zone,
                       self.w // self.zone, c_h)  # [b,s,s,h,t,h/s,w/s,per] [0,1,2,3,4,5,6,7]
        att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, n, c)
        output = self.output_linear(att)
        return output


class SummariseMultiHeadedAttention(nn.Module):
    def __init__(self, tokensize, d_model, head, p=0.1):
        super(SummariseMultiHeadedAttention, self).__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head
        self.h, self.w = tokensize
        sr_ratio = 2
        self.sr = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio, padding=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, t, h, w):
        bt, n, c = x.size()
        b = bt // t
        c_h = c // self.head
        query = self.query_embedding(x)  # [bt, n, c]
        # k,v summurize
        x_ = x.permute(0, 2, 1).reshape(bt, c, h, w)
        x_ = self.sr(x_).reshape(bt, c, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        key = self.key_embedding(x_)
        value = self.value_embedding(x_)
        zone = 1
        h = w = int(key.size()[1] ** 0.5)
        key = key.view(b, t, zone, h // zone, zone, w // zone, self.head,
                       c_h)  # [b,t,s,h/s,s,w/s,h,per]
        key = key.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, zone ** 2, self.head, -1,
                                                                    c_h)  # [b,s*s,h,t*h/s*w/s,per]
        query = query.view(b, t, zone, self.h // zone, zone, self.w // zone, self.head, c_h)
        query = query.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, zone ** 2, self.head, -1, c_h)
        value = value.view(b, t, zone, h // zone, zone, w // zone, self.head, c_h)
        value = value.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(b, zone ** 2, self.head, -1, c_h)
        att, _ = self.attention(query, key, value)
        att = att.view(b, zone, zone, self.head, t, self.h // zone,
                       self.w // zone, c_h)  # [b,s,s,h,t,h/s,w/s,per] [0,1,2,3,4,5,6,7]
        att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, n, c)
        output = self.output_linear(att)
        return output


class MTTLayer(nn.Module):
    def __init__(self, tokensize, hidden=96, num_head=4, dropout=0., zone=2):
        super().__init__()
        self.local_attention = WindowMultiHeadedAttention(tokensize=tokensize, d_model=hidden, head=num_head, p=dropout,
                                                          zone=zone)
        self.global_attention = SummariseMultiHeadedAttention(tokensize=tokensize, d_model=hidden, head=num_head,
                                                              p=dropout)

        self.ffn1 = FeedForward(hidden, p=dropout)
        self.ffn2 = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.norm4 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        x, t = input['x'], input['t']
        H, W = input['h'], input['w']
        x = x + self.local_attention(self.norm1(x), t=t)
        x = x + self.ffn1(self.norm2(x))
        x = x + self.global_attention(self.norm3(x), t=t, h=H, w=W)
        x = x + self.ffn2(self.norm4(x))
        return {'x': x, 't': t, 'h': H, 'w': W}


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CTDAttention(nn.Module):
    def __init__(self, hidden_dim=96, cross_scale=True):
        super(CTDAttention, self).__init__()
        self.early_stage = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.early_ca = ChannelAttention(hidden_dim)
        if cross_scale:
            self.late_stage = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )

    def forward(self, x, last_stage_feat=None):
        b, c, h, w = x.size()
        x_early = self.early_stage(x)
        x_early = x_early * self.early_ca(x_early)
        if last_stage_feat != None:
            last_stage_feat = F.interpolate(last_stage_feat, size=(h, w), mode='bilinear', align_corners=True)
            x_early = x_early + last_stage_feat
            x_early = self.late_stage(x_early)  # [b,c,h,w]
        return x_early


class CTDLayer(nn.Module):
    def __init__(self, hidden=96, dropout=0.0, cross_scale=True):
        super(CTDLayer, self).__init__()
        self.attention = CTDAttention(hidden_dim=hidden, cross_scale=cross_scale)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, last_stage_feat=None):
        b, c, h, w = x.size()
        x = x + self.attention(x, last_stage_feat)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = x + self.ffn(self.norm2(x))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return x


class CTD(nn.Module):
    def __init__(self, num_layers=4, hidden=96, cross_scale=True):
        super(CTD, self).__init__()
        layer = CTDLayer(hidden=hidden, dropout=0.0, cross_scale=cross_scale)
        self.layers = _get_clones(layer, num_layers)

    def forward(self, x, last_stage_feat=None):
        for i, enc_layer in enumerate(self.layers):
            x = enc_layer(x, last_stage_feat)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=4, hidden=96):
        super(TransformerDecoder, self).__init__()
        self.stage1 = CTD(num_layers=num_layers, hidden=hidden, cross_scale=False)
        self.stage2 = CTD(num_layers=num_layers, hidden=hidden, cross_scale=True)
        self.stage3 = CTD(num_layers=num_layers, hidden=hidden, cross_scale=True)
        self.stage4 = CTD(num_layers=num_layers, hidden=hidden, cross_scale=True)

    def forward(self, x4, x3, x2, x1):
        results = []
        x1_s = self.stage1(x1)
        results.append(x1_s)
        x2_s = self.stage2(x2, x1_s)
        results.append(x2_s)
        x3_s = self.stage3(x3, x2_s)
        results.append(x3_s)
        x4_s = self.stage4(x4, x3_s)
        results.append(x4_s)
        return results


class CoSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CoSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)

    def forward(self, img, flow):
        x = torch.cat([torch.mean(img, dim=1, keepdim=True),
                       torch.max(img, dim=1, keepdim=True)[0],
                       torch.mean(flow, dim=1, keepdim=True),
                       torch.max(flow, dim=1, keepdim=True)[0]], dim=1)
        x = self.conv1(x)
        return x


class CoChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(CoChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes * 2, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.in_planes = in_planes

    def forward(self, img, flow):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(torch.cat([img, flow], 1)))))  # B 2C 1 1
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(torch.cat([img, flow], 1)))))
        out = avg_out + max_out
        return out


class CoAttention(nn.Module):
    def __init__(self, inplanes, ratio=2):
        super(CoAttention, self).__init__()
        self.channel = CoChannelAttention(inplanes, ratio=ratio)
        self.spatial = CoSpatialAttention()

    def forward(self, spatial, temporal):
        ca = self.channel(spatial, temporal)
        pa = self.spatial(spatial, temporal)
        return ca, pa


class BFM(nn.Module):
    def __init__(self, in_dim, hidden_dim=96):
        super(BFM, self).__init__()
        self.r_down = nn.Sequential(ConvBNReLU(in_dim, hidden_dim, 3, 1, 1))
        self.f_down = nn.Sequential(ConvBNReLU(in_dim, hidden_dim, 3, 1, 1))
        # gate
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gate_conv = nn.Sequential(
            ConvBNReLU(2 * hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 2, 3, 1, 1)
        )
        # co-attention
        self.attn_fusion = CoAttention(hidden_dim, ratio=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, r, f):
        r = self.r_down(r)
        f = self.f_down(f)
        # gates
        gate = self.gate_conv(torch.cat([r, f], dim=1))
        gate = self.gap(self.sigmoid(gate))
        gate_r = gate[:, 0:1, ...] * r
        gate_f = gate[:, 1:2, ...] * f
        ca, pa = self.attn_fusion(gate_r, gate_f)
        a = self.sigmoid(ca + pa)
        gate_r = gate_r * a + gate_r
        gate_f = gate_f * (1 - a) + gate_f
        x = gate_r + gate_f
        return x


class MTNet(nn.Module):
    def __init__(self,
                 config=None):
        super(MTNet, self).__init__()
        self.config = config
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.frame_nums = config['train_frames']
        self.inference_frame_nums = config['clip_length']
        self.backbone = build_encoder(name=config['encoder_name'])
        self.hidden_dim = config['proj_dim']
        # BFM
        self.fusion1 = BFM(in_dim=self.in_channels[0], hidden_dim=self.hidden_dim)
        self.fusion2 = BFM(in_dim=self.in_channels[1], hidden_dim=self.hidden_dim)
        self.fusion3 = BFM(in_dim=self.in_channels[2], hidden_dim=self.hidden_dim)
        self.fusion4 = BFM(in_dim=self.in_channels[3], hidden_dim=self.hidden_dim)
        # MTT
        self.mtt_layers = config['mtt_layers']
        self.head = config['head']
        self.dropout = config['dropout']
        self.zone_size = config['zone_size']
        # CTD
        self.ctd_layers = config['ctd_layers']
        blocks3 = []
        blocks4 = []
        for _ in range(self.mtt_layers):
            blocks3.append(
                MTTLayer(tokensize=(32, 32), hidden=self.hidden_dim, num_head=self.head, dropout=self.dropout,
                         zone=self.zone_size))
            blocks4.append(
                MTTLayer(tokensize=(16, 16), hidden=self.hidden_dim, num_head=self.head, dropout=self.dropout,
                         zone=self.zone_size))
        self.transformer3 = nn.Sequential(*blocks3)
        self.transformer4 = nn.Sequential(*blocks4)
        # CTD
        self.decoder = TransformerDecoder(num_layers=self.ctd_layers, hidden=self.hidden_dim)
        self.fpn = FPN(self.out_channels)
        self.cls2 = nn.Sequential(nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(self.out_channels[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_channels[0], 1, kernel_size=1))
        self.cls3 = nn.Sequential(nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(self.out_channels[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_channels[0], 1, kernel_size=1))
        self.cls4 = nn.Sequential(nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(self.out_channels[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_channels[0], 1, kernel_size=1))
        self.cls5 = nn.Sequential(nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=3, padding=1),
                                  nn.BatchNorm2d(self.out_channels[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(self.out_channels[0], 1, kernel_size=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, imgs, flows):
        # images has shape: [b, t, c, h, w]
        # prepare input
        batch_size = self.config['batch_size'] if self.training else 1
        imgs = imgs.flatten(0, 1)
        flows = flows.flatten(0, 1)
        # feature extract
        img_conv1, img_conv2, img_conv3, img_conv4 = self.backbone(imgs)
        flow_conv1, flow_conv2, flow_conv3, flow_conv4 = self.backbone(flows)
        # multi-modal fusion
        x_conv1 = self.fusion1(img_conv1, flow_conv1)
        x_conv2 = self.fusion2(img_conv2, flow_conv2)
        x_conv3 = self.fusion3(img_conv3, flow_conv3)
        x_conv4 = self.fusion4(img_conv4, flow_conv4)
        # mix temporal transformer
        t = x_conv1.size()[0] // batch_size
        _, _, h, w = x_conv3.size()
        x_conv3 = x_conv3.flatten(2).permute(0, 2, 1).contiguous()
        x_conv3 = self.transformer3({'x': x_conv3, 't': t, 'h': 32, 'w': 32})['x']
        x_conv3 = x_conv3.permute(0, 2, 1).contiguous().view(-1, 96, h, w)  #
        x_conv4 = x_conv4.flatten(2).permute(0, 2, 1).contiguous()
        x_conv4 = self.transformer4({'x': x_conv4, 't': t, 'h': 16, 'w': 16})['x']
        x_conv4 = x_conv4.permute(0, 2, 1).contiguous().view(-1, 96, h // 2, w // 2)
        # cascaded transformer decoder
        p5, p4, p3, p2 = self.decoder(x_conv1, x_conv2, x_conv3, x_conv4)
        # segmentation
        p2, p3, p4, p5 = self.fpn(p2, p3, p4, p5)
        clip_preds_2 = self.cls2(p2)
        clip_preds_3 = self.cls3(p3)
        clip_preds_4 = self.cls4(p4)
        clip_preds_5 = self.cls5(p5)
        # upsamples
        clip_preds_2 = F.interpolate(clip_preds_2, imgs.shape[2:], mode='bilinear', align_corners=True)
        clip_preds_3 = F.interpolate(clip_preds_3, imgs.shape[2:], mode='bilinear', align_corners=True)
        clip_preds_4 = F.interpolate(clip_preds_4, imgs.shape[2:], mode='bilinear', align_corners=True)
        clip_preds_5 = F.interpolate(clip_preds_5, imgs.shape[2:], mode='bilinear', align_corners=True)
        if self.training:
            return clip_preds_2, clip_preds_3, clip_preds_4, clip_preds_5
        else:
            return self.sigmoid(clip_preds_2)

    def clip_inference(self, clip_length, imgs, flows, eval_fps=False):
        video_len = imgs.size()[1]
        this_video_time = 0
        if video_len > clip_length:
            num_clips = math.ceil(video_len / clip_length)
            masks_list = []
            for c in range(num_clips - 1):
                start_idx = c * clip_length
                end_idx = (c + 1) * clip_length
                clip_images = imgs[:, start_idx:end_idx, ...]
                clip_flows = flows[:, start_idx:end_idx, ...]
                start_time = time.time()
                clip_output = self.forward(clip_images, clip_flows)
                this_video_time += time.time() - start_time
                clip_output = clip_output.cpu().detach()
                masks_list.append(clip_output)
            start_time = time.time()
            masks_list.append(self.forward(imgs[:, (num_clips - 1) * clip_length:, ...],
                                           flows[:, (num_clips - 1) * clip_length:, ...]).cpu().detach())

            this_video_time += time.time() - start_time
            output = torch.cat(masks_list, dim=0)
        else:
            output = self.forward(imgs, flows).cpu().detach()
        if eval_fps:
            return output.cuda(), this_video_time
        else:
            return output.cuda()
