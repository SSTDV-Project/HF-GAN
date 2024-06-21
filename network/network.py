import torch
import math
from torch import nn
from einops import rearrange
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
 
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    
    def __init__(self, dim, dim_out, blocks, groups=8):
        super().__init__()
        self.blocks = blocks

        self.block1 = Block(dim, dim_out, groups=groups)
        if blocks:
            self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        if self.blocks:
            h = self.block2(h)
        return h + self.res_conv(x)

class UpSample(nn.Module):
    """
    ## Up-sampling layer
    """
    def __init__(self, channels):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)
    
class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """
    def __init__(self, channels):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Add padding
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        # Apply convolution
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, init_dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), blocks=True):
        super(Encoder, self).__init__()
        self.conv_initial = nn.Conv2d(num_inputs, init_dim, 3, padding=1)
        
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, blocks),
                        DownSample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            ) 

    def forward(self, x):
        x = self.conv_initial(x)

        h = []
        # downsample
        for down in self.downs:
            block, downsample = down
            x = block(x)
            h.append(x)
            x = downsample(x)

        return x, h
    
class Decoder(nn.Module):
    def __init__(self, init_dim=64, num_outputs=1, dim_mults=(1, 2, 4, 8, 10), skip=True, blocks=True, skip_multiplier=2):
        super(Decoder, self).__init__()
        self.skip = skip
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            if skip:
                dim_skip = int(dim_out*skip_multiplier)
            else:
                dim_skip = dim_out
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_skip, dim_in, blocks),
                        UpSample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )


        self.conv_final = nn.Sequential(
            ResnetBlock(init_dim, init_dim, blocks), nn.GroupNorm(8, init_dim), nn.SiLU(), nn.Conv2d(init_dim, num_outputs, 1)
        )

    def forward(self, x, h):
        # upsample
        for n, up in enumerate(self.ups):
            block, upsample = up
            if self.skip:
                x = torch.cat((x, h[::-1][n]), dim=1)
            x = block(x)
            x = upsample(x)

        return self.conv_final(x)
    
class CHattnblock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid())

    def forward(self, x):
        w = self.attn(x)
        # print(w.shape)
        return w
    
class HFEncoder(nn.Module):
    def __init__(self, dim=64, num_inputs=4, dim_mults=(1, 2, 4, 8, 10), n_layers=2, blocks=True, n_tokens=0):
        super().__init__()
        self.num_inputs = num_inputs
        self.encoder_early = Encoder(dim, num_inputs, dim_mults, blocks)
        self.encoder_middles = nn.ModuleList([Encoder(dim, 1, dim_mults, blocks) for i in range(num_inputs)])
        self.attn_blocks = nn.ModuleList([CHattnblock(dim*dim_mults[-1]) for i in range(num_inputs+1)])
        self.conv1 = nn.Conv2d(dim*dim_mults[-1]*2, dim*dim_mults[-1], 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, input_modals, train_mode=False):
        x_early, h_early = self.encoder_early(x)
        x_fusion_s = torch.zeros_like(x_early, device=x_early.device)
        x_fusion_h = torch.zeros_like(x_early, device=x_early.device)
        h_fusion = [torch.zeros_like(h, device=h.device) for h in h_early]
        
        x_middles = []
        h_middles = []
        x_attns = []
        for i in range(self.num_inputs):
            x_middle, h_middle = self.encoder_middles[i](x[:,i:i+1,:])
            x_middles.append(x_middle)
            h_middles.append(h_middle)
            x_attns.append(self.attn_blocks[i](x_middle))
        x_attns.append(self.attn_blocks[-1](x_early))

        for n, modals in enumerate(input_modals):
            x_att = []
            x_feat = []
            for i in modals:
                for h_fusion_feat, h_middle_feat in zip(h_fusion, h_middles[i]):
                    h_fusion_feat[n,:] += h_middle_feat[n,:] / len(modals)
                x_att.append(x_attns[i][n:n+1,:])
                x_feat.append(x_middles[i][n,:])
            if len(modals) != 1:
                x_att = torch.concat(x_att, dim=0)
                for idx, i in enumerate(modals):
                    x_fusion_s[n,:] += x_middles[i][n,:]*x_att[idx,0,:]
                x_fusion_s[n,:] += x_early[n,:]*x_attns[-1][n,:]
                
                x_att = self.softmax(x_att)
                for idx, i in enumerate(modals):
                    x_fusion_h[n,:] += x_middles[i][n,:]*x_att[idx,0,:]
                x_fusion_h[n,:] += x_early[n,:]
            else:
                for idx, i in enumerate(modals):
                    x_fusion_s[n,:] = x_middles[i][n,:]*x_att[0][0,:]
                    x_fusion_h[n,:] = x_middles[i][n,:]

        x_fusion = self.conv1(torch.concat((x_fusion_s, x_fusion_h), dim=1))
        if train_mode:
            idx_1ch = x.shape[0] // 2
            x = x_fusion

            h_combination = [f_e[0:idx_1ch,:] + f_fusion[0:idx_1ch,:] for f_e, f_fusion in zip(h_early, h_fusion)]
            h_1ch = [f_fusion[idx_1ch:,:] for f_fusion in h_fusion]
            h = [torch.cat([f_comb, f_1ch], dim=0) for f_comb, f_1ch in zip(h_combination, h_1ch)]
        else:
            x = x_fusion

            h = []
            for f_early, f_fusion in zip(h_early, h_fusion):
                f = []
                for n, modals in enumerate(input_modals):
                    if len(modals) == 1:
                        f.append(f_fusion[n:n+1,:])
                    else:
                        f_sum = f_early[n:n+1,:] + f_fusion[n:n+1,:]
                        f.append(f_sum)
                h.append(torch.cat(f, dim=0))
            
        return x, h
   
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, dim, patch_size, n_tokens):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=dim,
                                       out_channels=dim,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        #B,C,W,H = x.shape()
        x = self.patch_embeddings(x)
        x = x.flatten(2,3)
        h = x.permute(0,2,1)

        embeddings = h + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act_fn =  nn.SiLU()
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    
# Attention module
class Attention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        self.attn = Attention(hidden_size, n_heads)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        if time.dtype == torch.int64:
            return embeddings
        else:
            return embeddings.type(time.dtype)

class ModalityInfuser(nn.Module):
    def __init__(self, hidden_size, patch_size, n_tokens, n_layers, n_heads, modality_embed):
        super().__init__()
        self.modality_embed = modality_embed
        #n_tokens = int((240/(2**n_downs)/patch_size)**2)
        self.modality_embedding = nn.Sequential(
                SinusoidalPositionEmbeddings(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
        
        self.embedding =Embeddings(hidden_size, patch_size, n_tokens)
        self.layers = nn.ModuleList([])
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(hidden_size, n_heads)
            )

    def forward(self, x, m):
        B,C,W,H =x.shape
        h = self.embedding(x)
        if self.modality_embed:
            m = self.modality_embedding(m)
            h = rearrange(m, "b c -> b 1 c") + h

        for layer_block in self.layers:
            h = layer_block(h)
        h = self.encoder_norm(h)
        h = h.permute(0,2,1).contiguous().view(B,C,W,H)
        return h

class HFGAN(nn.Module):
    def __init__(self, dim, num_inputs, num_outputs, dim_mults, n_layers, skip, blocks, image_size=240):
        super().__init__()
        patch_size=1
        n_tokens = int((image_size/(2**(len(dim_mults)-1))/patch_size)**2)
        self.encoder = HFEncoder(dim, num_inputs, dim_mults, n_layers, blocks, n_tokens=n_tokens)
        self.decoder = Decoder(dim, num_outputs, dim_mults, skip, blocks)
        self.middle = ModalityInfuser(hidden_size=dim*dim_mults[-1],  patch_size=1, n_tokens=n_tokens, n_layers=n_layers, n_heads=16, modality_embed=True)

#PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
class Discriminator(nn.Module):
    def __init__(self, channels=1, num_filters_last=32, n_layers=3, n_classes=4, ixi=False):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.GroupNorm(8, num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.model = nn.Sequential(*layers)
        self.final = nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1, bias=False)
        if ixi:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 31, bias=False)
        else:
            self.classifier = nn.Conv2d(num_filters_last * num_filters_mult, n_classes, 29, bias=False)

    def forward(self, x):
        x = self.model(x)
        logits = self.final(x)
        labels = self.classifier(x)
        return logits, labels.view(labels.size(0), labels.size(1))