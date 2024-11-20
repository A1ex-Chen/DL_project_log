import torch.nn as nn
import torch

# L2 Distance squared
l2_dist = lambda x, y: (x - y) ** 2


class LayerNorm(nn.LayerNorm):



class StackLayerNorm(nn.Module):



class GatedConv2d(nn.Module):




class GatedPixelCNN(nn.Module):
    """
    The following Gated PixelCNN is taken from class material given on Piazza
    """



#class MaskedConv2d(nn.Conv2d):
class MaskedConv2d(nn.Module):
    """
    Class extending nn.Conv2d to use masks.
    """


        # return super().forward(x)


class VectorQuantizer(nn.Module):



class ResidualBlock(nn.Module):



class Encoder(nn.Module):



class Decoder(nn.Module):



class VQVAE(nn.Module):
        # self.pixelcnn_prior = GatedPixelCNN(K=K).to(device)
        # self.pixelcnn_loss_fct = nn.CrossEntropyLoss()



    """
    def get_pixelcnn_prior_loss(self, x, output):
        q, logit_probs = output
        return self.pixelcnn_loss_fct(logit_probs, q)

    def get_vae_loss(self, x, output):
        N, C, H, W = x.shape
        x_reconstructed = output
        z_e = self.z_e
        z_q = self.z_q

        reconstruction_loss = l2_dist(x, x_reconstructed).sum() / (N * H * W * C)
        vq_loss = l2_dist(z_e.detach(), z_q).sum() / (N * H * W * C)
        commitment_loss = l2_dist(z_e, z_q.detach()).sum() / (N * H * W * C)

        return reconstruction_loss + vq_loss + commitment_loss
    """