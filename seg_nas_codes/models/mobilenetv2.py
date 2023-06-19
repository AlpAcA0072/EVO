import torch

from ofa.utils import make_divisible, val2list, MyNetwork
from ofa.imagenet_classification.networks.proxyless_nets import ProxylessNASNets
from ofa.utils.layers import MBConvLayer, ConvLayer, IdentityLayer, LinearLayer, ResidualBlock


class MyProxylessNASNets(ProxylessNASNets):
    def __init__(self, features_only=False, **kwargs):
        super(MyProxylessNASNets, self).__init__(**kwargs)
        self.features_only = features_only
        self.feature_dim = None

    def forward_features(self, x):
        features = []
        for block in self.blocks:
            y = x
            x = block(x)
            if y.size()[2:] != x.size()[2:]:
                features.append(y)

        features.append(x)
        return features[-3:]  # assumes 1/8, 1/16 and 1/32 features

    def forward(self, x):
        x = self.first_conv(x)

        if self.features_only:
            return self.forward_features(x)
        else:
            for block in self.blocks:
                x = block(x)
            if self.feature_mix_layer is not None:
                x = self.feature_mix_layer(x)
            x = self.global_avg_pool(x)
            x = self.classifier(x)
            return x

    @property
    def config(self):
        if self.feature_dim is None:
            # run a dummy forward pass to collect output for measuring feature dimensions
            x = torch.rand(1, 3, 224, 224)
            x = self.first_conv(x)
            features = self.forward_features(x)
            self.feature_dim = [v.size(1) for v in features]
        return {
            'name': ProxylessNASNets.__name__,
            'feature_dim': self.feature_dim,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }


class MobileNetV2(MyProxylessNASNets):
    def __init__(self, n_classes=1000, width_mult=1.0, bn_param=(0.1, 1e-3), dropout_rate=0.2,
                 ks=None, expand_ratio=None, depth_param=None, stage_width_list=None, features_only=False):

        ks = 3 if ks is None else ks
        expand_ratio = 6 if expand_ratio is None else expand_ratio

        input_channel = 32
        last_channel = 1280

        input_channel = make_divisible(input_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(last_channel * width_mult, MyNetwork.CHANNEL_DIVISIBLE) \
            if width_mult > 1.0 else last_channel

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [expand_ratio, 24, 2, 2],
            [expand_ratio, 32, 3, 2],
            [expand_ratio, 64, 4, 2],
            [expand_ratio, 96, 3, 1],
            [expand_ratio, 160, 3, 2],
            [expand_ratio, 320, 1, 1],
        ]

        if depth_param is not None:
            assert isinstance(depth_param, int)
            for i in range(1, len(inverted_residual_setting) - 1):
                inverted_residual_setting[i][2] = depth_param

        if stage_width_list is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = stage_width_list[i]

        ks = val2list(ks, sum([n for _, _, n, _ in inverted_residual_setting]) - 1)
        _pt = 0

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        # inverted residual blocks
        blocks = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                if t == 1:
                    kernel_size = 3
                else:
                    kernel_size = ks[_pt]
                    _pt += 1
                mobile_inverted_conv = MBConvLayer(
                    in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, stride=stride,
                    expand_ratio=t,
                )
                if stride == 1:
                    if input_channel == output_channel:
                        shortcut = IdentityLayer(input_channel, input_channel)
                    else:
                        shortcut = None
                else:
                    shortcut = None

                blocks.append(
                    ResidualBlock(mobile_inverted_conv, shortcut)
                )
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act',
        )

        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(MobileNetV2, self).__init__(
            features_only=features_only, first_conv=first_conv, blocks=blocks,
            feature_mix_layer=feature_mix_layer, classifier=classifier)

        # set bn param
        self.set_bn_param(*bn_param)


if __name__ == '__main__':

    data = torch.rand(1, 3, 224, 224)
    model = MobileNetV2(features_only=True)
    out = model(data)

    if type(out) == list:
        print([v.size() for v in out])
    else:
        print(out.size())

    print(model.config['feature_dim'])