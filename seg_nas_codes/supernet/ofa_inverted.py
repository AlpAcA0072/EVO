import copy
import random

from ofa.utils import make_divisible, val2list, MyNetwork
from ofa.imagenet_classification.elastic_nn.modules import DynamicMBConvLayer
from ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock

from models.mobilenetv2 import MyProxylessNASNets


class OFAInvertedBottleneckNets(MyProxylessNASNets):

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-3), dropout_rate=0.1, base_stage_width=None, width_mult=1.0,
                 ks_list=3, expand_ratio_list=6, depth_list=4, features_only=False):
        self.width_mult = width_mult
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)

        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        if base_stage_width == 'google':
            # MobileNetV2 Stage Width
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        else:
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]

        input_channel = make_divisible(base_stage_width[0] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        first_block_width = make_divisible(base_stage_width[1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

        # first conv layer
        first_conv = ConvLayer(
            3, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act'
        )
        # first block
        first_block_conv = MBConvLayer(
            in_channels=input_channel, out_channels=first_block_width, kernel_size=3, stride=1,
            expand_ratio=1, act_func='relu6',
        )
        first_block = ResidualBlock(first_block_conv, None)

        input_channel = first_block_width
        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]
        n_block_list = [max(self.depth_list)] * 5 + [1]

        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
            width_list.append(width)

        for width, n_block, s in zip(width_list, n_block_list, stride_stages):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicMBConvLayer(
                    in_channel_list=val2list(input_channel, 1), out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list, stride=stride, act_func='relu6',
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = ResidualBlock(mobile_inverted_conv, shortcut)

                blocks.append(mb_inverted_block)
                input_channel = output_channel
        # 1x1_conv before global average pooling
        feature_mix_layer = ConvLayer(
            input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6',
        )
        classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

        super(OFAInvertedBottleneckNets, self).__init__(
            features_only=features_only, first_conv=first_conv, blocks=blocks,
            feature_mix_layer=feature_mix_layer, classifier=classifier)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAInvertedBottleneckNets'

    def forward_features(self, x):
        features = []

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

            # assumes 1/8, 1/16 and 1/32 features
            if stage_id in [1, 3, 5]:
                features.append(x)

        # # feature_mix_layer
        # x = self.feature_mix_layer(x)
        # features.append(x)

        return features

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        if self.features_only:
            return self.forward_features(x)
        else:
            # blocks
            for stage_id, block_idx in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id]
                active_idx = block_idx[:depth]
                for idx in active_idx:
                    x = self.blocks[idx](x)

            # feature_mix_layer
            x = self.feature_mix_layer(x)
            x = x.mean(3).mean(2)

            x = self.classifier(x)
            return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFAInvertedBottleneckNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    @property
    def grouped_block_index(self):
        return self.block_group_info

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if '.mobile_inverted_conv.' in key:
                new_key = key.replace('.mobile_inverted_conv.', '.conv.')
            else:
                new_key = key
            if new_key in model_dict:
                pass
            elif '.bn.bn.' in new_key:
                new_key = new_key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in new_key:
                new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in new_key:
                new_key = new_key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in new_key:
                new_key = new_key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in new_key:
                new_key = new_key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in new_key:
                new_key = new_key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(new_key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = state_dict[key]
        super(OFAInvertedBottleneckNets, self).load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_max_net(self):
        self.set_active_subnet(ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list))

    def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
        ks = val2list(ks, len(self.blocks) - 1)
        expand_ratio = val2list(e, len(self.blocks) - 1)
        depth = val2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.conv.active_kernel_size = k
            if e is not None:
                block.conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']

        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']

        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # print(ks_candidates)
        # print(expand_candidates)
        # print(depth_candidates)

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        depth_setting[-1] = 1
        self.set_active_subnet(ks_setting, expand_setting, depth_setting)

        return {
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]
        feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        classifier = copy.deepcopy(self.classifier)

        input_channel = blocks[0].conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(ResidualBlock(
                    self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].conv.out_channels
            blocks += stage_blocks

        _subnet = MyProxylessNASNets(features_only=self.features_only, first_conv=first_conv,
                                     blocks=blocks, feature_mix_layer=feature_mix_layer, classifier=classifier)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    def get_active_net_config(self):
        first_conv_config = self.first_conv.config
        first_block_config = self.blocks[0].config
        feature_mix_layer_config = self.feature_mix_layer.config
        classifier_config = self.classifier.config

        block_config_list = [first_block_config]
        input_channel = first_block_config['conv']['out_channels']
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append({
                    'name': ResidualBlock.__name__,
                    'conv': self.blocks[idx].conv.get_active_subnet_config(input_channel),
                    'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
                })
                try:
                    input_channel = self.blocks[idx].conv.active_out_channel
                except Exception:
                    input_channel = self.blocks[idx].conv.out_channels
            block_config_list += stage_blocks

        return {
            'name': MyProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'first_conv': first_conv_config,
            'blocks': block_config_list,
            'feature_mix_layer': feature_mix_layer_config,
            'classifier': classifier_config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.conv.re_organize_middle_weights(expand_ratio_stage)


if __name__ == '__main__':
    import torch

    data = torch.rand(1, 3, 224, 224)

    # # inverted bottleneck block ofa backbone
    ofa_network = OFAInvertedBottleneckNets(
        dropout_rate=0, width_mult=1.0, ks_list=3, expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
        features_only=True)

    # set subnet sampling constraints
    ofa_network.set_constraint([2, 3, 4], constraint_type='depth')
    ofa_network.set_constraint([3, 4, 6], constraint_type='expand_ratio')
    # ofa_network.set_constraint([1, 2], constraint_type='width_mult')

    # ofa_network.load_state_dict(torch.load('../pretrained/backbone/inverted.pth', map_location='cpu'))
    arch_config = ofa_network.sample_active_subnet()
    print(arch_config)

    # Manually setting sub-network to minimal bound subnet
    # ofa_network.set_active_subnet(d=[0, 0, 0, 0], e=[1.0] * 16, w=[0] * 5)
    # print(arch_config)

    subnet = ofa_network.get_active_subnet()
    # print(subnet)
    out = subnet(data)
    if type(out) == list:
        print([v.size() for v in out])
    else:
        print(out.size())

    print(subnet.config['feature_dim'])
    # flops = profile_macs(ofa_network, data) / 1e6
    # print(flops)

    # Randomly sample sub-networks from OFA network
    # random_subnet = ofa_network.get_active_subnet(preserve_weight=True)
    # print(random_subnet)

    # weights = torch.load("pretrained/resnet12d.pth", map_location='cpu')
    # random_subnet.load_state_dict(weights)
    #
    # random_subnet.eval()
    # out = random_subnet(data)

    # if type(out) == list:
    #     print([v.size() for v in out])
    # else:
    #     print(out.size())
    # print(random_subnet.config)

    # # for v in out[1]:
    # #     print(v.size())
    # flops = profile_macs(random_subnet, data) / 1e6
    # print(flops)