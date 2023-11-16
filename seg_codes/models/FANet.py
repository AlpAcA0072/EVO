import torch
import torch.nn as nn
import torch.nn.functional as F
# from dcn_v2 import DCN as dcn_v2
from DCNv2.dcn_v2 import DCN as dcn_v2
from nets.stdcnet import STDCNet1446, STDCNet813
from nets.resnet import *
from nets.dfnet import *
from nets.mobilenetv3 import *
from nets.efficientnets import *
from utils import pretrained_model_names
from .ops import ConvBNReLU, BatchNorm2d, BiSeNetOutput


class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureSelectionModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat = self.conv(x)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureAlign(nn.Module):
    def __init__(self, in_nc=128, out_nc=128):
        super(FeatureAlign, self).__init__()
        self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.offset = ConvBNReLU(out_nc * 2, out_nc, 1, 1, 0)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc//2, 3, stride=1, padding=1, dilation=1, deformable_groups=8, extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        self.fsm_cat = ConvBNReLU(out_nc//2, out_nc, 1, 1, 0)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

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

        return feat, feat_arm


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
            if args.machine == 'dgx2':
                pretrained_weights = '/raid/huangsh/codes/PreTrained_Weights/{}'.format(pretrained_model_names[args.backbone])
            elif args.machine == 'inspur':
                pretrained_weights = '/shihuahuang/codes/PreTrained_Weights/{}'.format(pretrained_model_names[args.backbone])
            elif args.machine == 'cseadmin':
                pretrained_weights = '/home/cseadmin/huangsh/codes/PreTrained_Weights//{}'.format(pretrained_model_names[args.backbone])
            print("     ### Loading Pretrained Weights From {} ###      ".format(pretrained_weights))
            self.backbone.load_state_dict(torch.load(pretrained_weights), strict=False)

    def forward(self, x):
        if 'STDCNet' in self.backbone_name or 'mobilenetv3' in self.backbone_name:
            feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        else:
            feat8, feat16, feat32 = self.backbone(x)
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class Context(nn.Module):   #
    def __init__(self, in_nc=128, out_nc=128):
        super(Context, self).__init__()
        self.fsm = FeatureSelectionModule(in_nc, out_nc)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def forward(self, feat32):
        feat32 = self.fsm(feat32)     # 0~1 * feats
        return feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


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
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_res16, feat_res32 = self.cp(x)
        feat32_context = self.context(feat_res32)
        feat16_align, feat16_detail = self.align16(feat_res16, feat32_context)
        feat8_align, feat8_detail = self.align8(feat_res8, feat16_align)
        feat_out = self.conv_out(feat8_align)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        if self.output_aux:
            aux_out32 = F.interpolate(self.conv_out32(feat32_context), (H, W), mode='bilinear', align_corners=True)
            aux_out16 = F.interpolate(self.conv_out16(feat16_align), (H, W), mode='bilinear', align_corners=True)
            return feat_out, aux_out16, aux_out32, None
        else:
            return feat_out, None, None, None

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureAlign, Context, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


if __name__ == "__main__":
    net = FANet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')


