import torch
import torch.nn as nn
from mmpretrain.registry import MODELS
from mmpretrain.models.backbones import VisionTransformer
from mmpretrain.models.utils import resize_pos_embed
from mmpretrain.models.utils import build_norm_layer

@MODELS.register_module()
class PromptedViT(VisionTransformer):
    '''
    
    prompt_length (int):
    deep_prompt (bool):
    prompt_init (str):
    '''

    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap', 'avg_all', 'avg_prompt', 'avg_prompt_clstoken', 'avg_three'}
    # 'avg_all' : avg of 'prompt' & 'cls_token' & 'featmap'
    # 'avg_prompt' avg of 'prompt'
    # 'avg_prompt_clstoken' avg of 'cls_token' and 'prompt'
    # 'avg_three' avg of 'cls_token', avg(prompt) and avg(featmap)
    def __init__(self,
                 prompt_length = 1,#提示长度，用于对prompt张量初始化
                 deep_prompt = True, #当 deep_prompt 为 True 时，模型会在多个层之间使用深层提示。具体来说，对于每个模型层，都会使用相同长度的提示张量。这使得模型可以在不同的层次上获得不同的提示信息，从而更灵活地适应不同的特征层。
                 out_type='avg_all',
                 prompt_init: str = 'normal', #冒号后为类型注解，表示prompt_init为字符串类型
                 norm_cfg=dict(type='LN'), #将 Layer Normalization（LN）作为规范化的配置
                 *args, #接受不定数量的位置参数
                 **kwargs):#接受不定数量的关键字参数
        super().__init__(*args, out_type=out_type,  norm_cfg=norm_cfg, **kwargs)

        self.prompt_layers = len(self.layers) if deep_prompt else 1
        prompt = torch.empty(
            self.prompt_layers, prompt_length, self.embed_dims) #prompt创建了一个空的张量
        if prompt_init == 'uniform':
            nn.init.uniform_(prompt, -0.08, 0.08) #将prompt设置为每个元素为-0.08-0.08的均匀分布的张量
        elif prompt_init == 'zero':
            nn.init.zeros_(prompt) #将prompt设置为每个元素为0的张量
        elif prompt_init == 'kaiming':
            nn.init.kaiming_normal_(prompt)
        elif prompt_init == 'token':
            nn.init.zeros_(prompt)
            self.prompt_initialized = False
        else:
            nn.init.normal_(prompt, std=0.02) #正态初始化，标准差0.02
        self.prompt = nn.Parameter(prompt, requires_grad=True) #nn.Parameter 是一种特殊的张量类型，它被设计为模型参数（可学习的参数）
        self.prompt_length = prompt_length
        self.deep_prompt = deep_prompt

        if self.out_type in {'avg_featmap', 'avg_all', 'avg_prompt', 'avg_prompt_clstoken', 'avg_three'}:
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims) #build_norm_layer用以创建规范化层
        
        # freeze stages 
        self.frozen_stages = len(self.layers) #冻结所有层
        self._freeze_stages() #执行冻结

    def forward(self, x):
        B = x.shape[0] #获取输入张量 x 的批次大小（batch size）
        x, patch_resolution = self.patch_embed(x) #通过调用 self.patch_embed 方法，将输入张量 x 映射为一系列图像块。返回值 x 是映射后的张量，patch_resolution 是图像块的分辨率。

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1) #维度扩展，-1表示该维度上大小保持不变
            x = torch.cat((cls_token, x), dim=1) #沿着dim=1，即第二维度，拼接cls_token和x

        x = x + resize_pos_embed( #调整position embedding大小（常用于自注意力机制）以适应张量x
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x) #训练神经网络时随机丢弃神经元，减轻过拟合

        x = self.pre_norm(x) #归一化（通常为批归一化或层归一化）

        # reshape to [layers, batch, tokens, embed_dims]
        prompt = self.prompt.unsqueeze(1).expand(-1, x.shape[0], -1, -1)
        x = torch.cat(
                [x[:, :1, :], prompt[0, :, :, :], x[:, 1:, :]],
                dim=1)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            if self.deep_prompt and i != len(self.layers) - 1:
                x = torch.cat(
                        [x[:, :1, :], prompt[i, :, :, :], x[:, self.prompt_length + 1:, :]],
                        dim=1)

            # final_norm should be False here
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):#根据不同的输出类型（self.out_type）对模型的输出进行格式化
        if self.out_type == 'raw': #原始输出，直接返回未经处理的模型输出 x
            return x
        if self.out_type == 'cls_token': #返回模型输出中的第一个令牌（通常是类别令牌）
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(x[:, self.prompt_length+1:].mean(dim=1))     
        if self.out_type == 'avg_all':
            return self.ln2(x.mean(dim=1))  
        if self.out_type == 'avg_prompt':
            return self.ln2(x[:, 1:self.prompt_length+1].mean(dim=1))  
        if self.out_type == 'avg_prompt_clstoken':
            return self.ln2(x[:, :self.prompt_length+1].mean(dim=1))  
        if self.out_type == 'avg_three':
            avg_feat_token = x[:, self.prompt_length+1:].mean(dim=1)
            avg_prompt = x[:, 1:self.prompt_length + 1].mean(dim=1)
            cls_token = x[:, 0]
            return self.ln2( avg_feat_token + avg_prompt + cls_token )  
         
