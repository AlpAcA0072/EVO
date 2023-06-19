import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_v2 import DCN as dcn_v2
from nets.stdcnet_test import STDCNet1446, STDCNet813
from nets.resnet_test import *
from nets.mobilenetv3_test import *
from nets.dfnet_test import *
from nets.efficientnets_test import *
BatchNorm2d = nn.BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        # self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv(x)
        return feat


class FeatureAlign(nn.Module):
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign, self).__init__()
        self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.offset = ConvBNReLU(out_nc * 2, out_nc, 1, 1, 0)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc//2, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        self.fsm_cat = ConvBNReLU(out_nc//2, out_nc, 1, 1, 0)

    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='nearest')
        else:
            feat_up = feat_s
        feat_arm = self.fsm(feat_l)  # 0~1 * feats
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset]))  # [feat, offset]
        # fcat = torch.cat([feat_arm, feat_align], dim=1)
        feat = self.fsm_cat(feat_align) + feat_arm

        return feat


class ContextPath(nn.Module):
    def __init__(self, args):
        super(ContextPath, self).__init__()

        self.backbone_name = args.backbone
        if args.backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=args.pretrain_model, use_conv_last=args.use_conv_last)
        elif args.backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=args.pretrain_model, use_conv_last=args.use_conv_last)
        else:
            self.backbone = eval(args.backbone)(pretrained=False)

    def forward(self, x):
        if 'STDCNet' in self.backbone_name or 'mobilenet' in self.backbone_name:
            feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        else:
            feat8, feat16, feat32 = self.backbone(x)
        return feat8, feat16, feat32


class Context(nn.Module):   #
    def __init__(self, in_nc=128, out_nc=128):
        super(Context, self).__init__()
        self.fsm = FeatureSelectionModule(in_nc, out_nc)

    def forward(self, feat32):
        feat32 = self.fsm(feat32)     # 0~1 * feats
        return feat32


class FANet(nn.Module):
    def __init__(self, args):
        super(FANet, self).__init__()
        if 'STDCNet' in args.backbone:
            self.nc_8, self.nc_16, self.nc_32 = [256, 512, 1024]
        elif 'mobilenetv3' in args.backbone:
            self.nc_8, self.nc_16, self.nc_32 = [40, 112, 960]
        elif 'mobilenetv2' in args.backbone:
            self.nc_8, self.nc_16, self.nc_32 = [32, 96, 320]
        else:
            self.nc_8, self.nc_16, self.nc_32 = [128, 256, 512]
        self.out_nc = 128
        self.output_aux = args.output_aux

        self.cp = ContextPath(args)
        self.context = Context(in_nc=self.nc_32, out_nc=self.out_nc)
        self.align16 = FeatureAlign(in_nc=self.nc_16, out_nc=self.out_nc)
        self.align8 = FeatureAlign(in_nc=self.nc_8, out_nc=self.out_nc)
        self.conv_out = BiSeNetOutput(self.out_nc, self.out_nc//2, args.n_classes)
        self.conv_out16 = BiSeNetOutput(self.out_nc, self.out_nc//2, args.n_classes)
        self.conv_out32 = BiSeNetOutput(self.out_nc, self.out_nc//2, args.n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_res16, feat_res32 = self.cp(x)
        feat32_context = self.context(feat_res32)
        feat16_align = self.align16(feat_res16, feat32_context)
        feat8_align = self.align8(feat_res8, feat16_align)
        feat_out = self.conv_out(feat8_align)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        if self.output_aux:
            aux_out32 = F.interpolate(self.conv_out32(feat32_context), (H, W), mode='bilinear', align_corners=True)
            aux_out16 = F.interpolate(self.conv_out16(feat16_align), (H, W), mode='bilinear', align_corners=True)
            return feat_out, aux_out16, aux_out32
        else:
            return feat_out, None, None


if __name__ == "__main__":
    net = FANet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')


