import torch
import torch.nn as nn
from collections import OrderedDict
from random import randint, shuffle
from random import random as rand

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class mim_decoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embed_dim = 512
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim//64, batch_first=True)
        self.cross_fushion = Transformer(width=self.embed_dim,layers=4,heads=self.embed_dim//64)
        self.decoder_norm1 = nn.LayerNorm(self.embed_dim)
        self.decoder_norm2 = nn.LayerNorm(self.embed_dim)
        self.mim_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', nn.LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, 512))])) # 3*384*96
        self.init_decoder_params()

    def init_decoder_params(self):
        scale = self.cross_fushion.width**-0.5
        proj_std = scale * ((2 * self.cross_fushion.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_fushion.width)**-0.5
        nn.init.normal_(self.self_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.self_attn.out_proj.weight, std=proj_std)
        for block in self.cross_fushion.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mim_head.fc.weight, std=proj_std)

    def forward(self, image_feats, text_feats):
        image_feats = image_feats.unsqueeze(1)
        text_feats = text_feats.unsqueeze(1)
        image_feats = self.self_attn(image_feats, 
                                     text_feats, 
                                     text_feats, 
                                     need_weights=False)[0]
        fushion_feats = self.decoder_norm1(image_feats)

        x = self.cross_fushion(fushion_feats)

        x = self.decoder_norm2(x)
        x = self.mim_head(x)
        x = x.squeeze(1)
        return x


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

        print("len(tokenizer.id2token): ", len(self.id2token), "  ----  cls_token_id: ", self.cls_token_id,
              "  ----  mask_token_id: ", self.mask_token_id, flush=True)

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i  # self.id2token[i]

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        # assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos



if __name__ == '__main__':
    image_feats = torch.randn(1, 512,512)
    text_feats = torch.randn(10, 512)
    model = mim_decoder()
    output = model(image_feats, text_feats)
    print(output.shape)