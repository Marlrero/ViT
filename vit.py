# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
# %%
img = Image.open('./cat.jpg')

fig = plt.figure()
plt.imshow(img)
# %%
transform = Compose([Resize((32, 32)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0)  # add batch dimension
x.shape
# %%
patch_size = 4 # pixel
# eniops를 쓰면 좋은 점 (자연어 방식으로 이렇게 reshape가 가능)
patches = rearrange(x, 'b c (h s1) (w s2) -> b c (h s1) (w s2)', \
                    s1=patch_size, s2=patch_size)
patches.shape
# %%
patch_size = 4 # pixel
# eniops를 쓰면 좋은 점 (자연어 방식으로 이렇게 reshape가 가능)
patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', \
                    s1=patch_size, s2=patch_size)
patches.shape
# %%
'''class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

print(x.shape)
print(PatchEmbedding()(x).shape)'''
# %%
'''class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
print(x.shape)
print(PatchEmbedding()(x).shape)'''
# %%
'''class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128):
        self.patch_size = patch_size
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        print('init cls_token:', self.cls_token.shape)
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.proj(x)
        print('init x:', x.shape)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        print('cls_tokens:', cls_tokens.shape)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        print('result x:', x.shape)
        return x

PatchEmbedding()(x)'''

# %%
'''class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, 
                 emb_size: int = 128, img_size: int = 32):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
        print(self.positions.shape)

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
    
PatchEmbedding()(x)'''
# %%
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, 
                 img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x
# %%
device = torch.device("cuda")
x = torch.randn(16, 3, 224, 224).to(device)  # patch_size, in_channels, img_size, img_size
patch_embedding = PatchEmbedding().to(device)
patch_output = patch_embedding(x)
print('[batch, 1+num of patches, emb_size] = ', patch_output.shape)
# %%
'''class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 512, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # key, queries, values를 헤드의 개수로 분리
        # batch, num_heads, sequence length, embedding size
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
       
        # 마지막 축을 기준으로 sum -> queries와 keys 대상 수행
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        
        # 세번째 축을 기준으로 sum
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)") # batch, num_heads, values_len, embedding size
        out = self.projection(out)
        return out'''

# %%
# 쿼리, 키, 값을 한 번에 연산하는 버전
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 쿼리, 키, 값을 한꺼번에 행렬로 계산
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # key, queries, values를 헤드의 개수로 분리
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # 마지막 축을 기준으로 sum -> queries와 keys 대상 수행
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # 세번째 축을 기준으로 sum
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
# %%
MHA = MultiHeadAttention().to(device)
MHA_output = MHA(patch_output)
print(MHA_output.shape)
# %%
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
# %%
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
# %%
x = torch.randn(16, 1, 128).to(device)
model = FeedForwardBlock(128).to(device)
output = model(x)
print(output.shape)