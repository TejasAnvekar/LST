import torch
from torch import nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F


@torch.no_grad()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Mixerblock(nn.Module):
    def __init__(self, dim, num_patchs, token_dim, channel_dim):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patchs, token_dim),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim, channel_dim))

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'image dim must be divisible by patch size'
        self.num_patch = (image_size//patch_size)**2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(Mixerblock(
                dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))


    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer in self.mixer_blocks:
            x = mixer(x)

        x = self.layer_norm(x)
        attention = x
        x = x.mean(dim=1)

        return self.mlp_head(x), attention


def test():
    inp = torch.randn((1, 1, 28, 28)).cuda()

    model = MLPMixer(in_channels=1, image_size=28, patch_size=14, num_classes=10,
                     dim=196, depth=1, token_dim=2, channel_dim=196//4).cuda()
    print(model)

    out, _ = model(inp)

    print(out.shape)


class decoder(nn.Module):
    def __init__(self, inp=10, layers=[2000, 500, 500, 784]):
        super().__init__()
        x = []
        for layer in layers:
            x.append(nn.Linear(inp, layer))
            if layer == layers[-1]:
                x.append(nn.Sigmoid())
            else:
                x.append(nn.ReLU())

            inp = layer
        self.dec = nn.Sequential(*x)



    def forward(self, x):
        return self.dec(x)


class encoder(nn.Module):
    def __init__(self, inp=100, layers=[2000, 10]):
        super().__init__()
        x = []
        for layer in layers:
            x.append(nn.Linear(inp, layer))
            if layer != layers[-1]:
                x.append(nn.ReLU())

            inp = layer
        self.enc = nn.Sequential(*x)



    def forward(self, x):
        return self.enc(x)


def test_dec():
    inp = torch.randn((1, 10)).cuda()
    model = decoder().cuda()
    out = model(inp)
    print(model)
    print(out.shape)


class LST_AE(nn.Module):
    def __init__(self, in_channels=1, dim=28*28//4, num_classes=10, patch_size=14, image_size=28, depth=1, token_dim=4//2, channel_dim=(28*28)//(4*4), enc_layers=[2000, 10], dec_layers=[2000, 500, 500, 28*28]):
        super().__init__()
        self.LST = MLPMixer(
            in_channels=in_channels, dim=dim, num_classes=num_classes, patch_size=patch_size, image_size=image_size, depth=depth, token_dim=token_dim, channel_dim=channel_dim
        )
        self.encoder = encoder(inp=num_classes, layers=enc_layers)
        self.decoder = decoder(inp=enc_layers[-1], layers=dec_layers)

    def forward(self, x):
        n, c, h, w = x.shape
        x_bar, attention = self.LST(x)
        z = self.encoder(x_bar)
        x_hat = self.decoder(z)
        x_hat = x_hat.reshape(n, c, h, w)
        return x_hat, z, attention


class LST_VAE(nn.Module):
    def __init__(self, in_channels=1, dim=28*28//4, num_classes=10, patch_size=14, image_size=28, depth=1, token_dim=4//2, channel_dim=(28*28)//(4*4), enc_layers=[2000, 10], dec_layers=[2000, 500, 500, 28*28]):
        super().__init__()
        self.LST = MLPMixer(
            in_channels=in_channels, dim=dim, num_classes=num_classes, patch_size=patch_size, image_size=image_size, depth=depth, token_dim=token_dim, channel_dim=channel_dim
        )
        self.encoder = encoder(inp=num_classes, layers=enc_layers)
        self.decoder = decoder(inp=enc_layers[-1], layers=dec_layers)
        self.mu = nn.Linear(enc_layers[-1],enc_layers[-1])
        self.logvar = nn.Linear(enc_layers[-1],enc_layers[-1])
        self.vloss = None


        nn.init.xavier_normal_(self.mu.weight.data)
        nn.init.xavier_normal_(self.logvar.weight.data)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)



    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu +(std*eps)

    def VAE_loss(self,recon,x,mu,logvar):
        BCE = F.binary_cross_entropy(recon,x,reduction='mean')
        KLD = torch.mean(-0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp(),dim=1),dim=0)
        return BCE + 0.05*KLD

    def forward(self, x):
        n, c, h, w = x.shape
        x_bar, attention = self.LST(x)
        z = self.encoder(x_bar)
        mu = self.mu(z)
        logvar = self.logvar(z)
        z_sampled = self.reparameterize(mu,logvar)
        x_hat = self.decoder(z_sampled)
        x_hat = x_hat.reshape(n, c, h, w)
        self.vloss = self.VAE_loss(x_hat,x,mu,logvar)
        return x_hat,z, attention



def test_LST_AE():
    inp = torch.randn((1, 1, 28, 28)).cuda()
    model = LST_AE().cuda()
    out, z, attention = model(inp)
    print(model)
    print(out.shape, z.shape, attention.shape)


# test_LST_AE()

class IDEC(nn.Module):
    def __init__(self,model,n_z, n_clusters=10, alpha=1, pretrain_path=""
                 ):
        super().__init__()

        self.alpha = alpha
        self.pretrain_path = pretrain_path

        self.ae = model

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters,n_z))
        torch.nn.init.kaiming_normal_(self.cluster_layer.data)


    def pretrain(self):
        if self.pretrain_path=="":
            print("Pretrain path not given")
        else:
            self.ae.load_state_dict(torch.load(self.pretrain_path)["weights"])
            print("loaded pretrained model")

    def forward(self,x):
        x_hat,z,attention = self.ae(x)
        q = 1.0/(1.0+torch.sum(torch.pow(z.unsqueeze(1)-self.cluster_layer,2),2)/self.alpha)
        q = q.pow((self.alpha+1.0)/2.0)
        q = (q.t()/torch.sum(q,1)).t()
        return x_hat,q,attention
        