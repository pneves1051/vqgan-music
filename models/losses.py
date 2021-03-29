import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spec_transforms import stft, spec, create_mel

# VQ_VAE losses

def latent_loss(codes, beta=0.25):
  #z_e, z_q = codes[0], codes[1]
  z_e, z_q = torch.split(codes, int(codes.shape[-1]/2), dim=-1)
  #print(z_e.norm(), z_q.norm())
  vq_loss=torch.mean((z_e.detach()-z_q)**2)
  commit_loss = torch.mean((z_e-z_q.detach())**2)
  latent_loss = vq_loss+beta*commit_loss
  return latent_loss

def spectral_loss(real, fake, n_fft=1024, hop_length=256):
  real_spec = spec(stft(real, n_fft = 1024, hop_length=256))
  fake_spec = spec(stft(fake, n_fft = 1024, hop_length=256))
  spectral_loss = torch.linalg.norm(real_spec.view(real_spec.shape[0], -1) 
                                    - fake_spec.view(fake_spec.shape[0], -1), ord='fro')
  return spectral_loss

def mel_loss(real, fake, n_fft=1024, hop_length=256, sample_rate=44100):
  real_mel,_ = create_mel(real, n_fft, hop_length, sample_rate=sample_rate)
  fake_mel,_ = create_mel(fake, n_fft, hop_length, sample_rate=sample_rate)
  mel_loss = torch.linalg.norm(real_mel.view(real_mel.shape[0], -1) 
                                    - fake_mel.view(fake_mel.shape[0], -1), ord='fro')
  return mel_loss

def vq_vae_loss(real, fake, codes, beta=0.25, spec=True):
  l2_loss = F.mse_loss(fake, real)
  lat_loss = latent_loss(codes, beta=beta)
  spec_loss = 0
  if spec:
    spec_loss = spectral_loss(fake, real)
  vqvae_loss = l2_loss + lat_loss + spec_loss
  loss_list = [l2_loss.item(), lat_loss.item(), spec_loss.item()]
  return vqvae_loss, loss_list

# GAN losses

# INCLUDE D OR G TO CALCULATE PREDS AND INTERPOLATION
def wgan_loss(discriminator, real, fake, d_real, d_fake, mode):
  if mode == 'd':
    d_loss = -(d_real.mean() - d_fake.mean())
    # Gradient penalty
    eps = torch.rand((d_real.shape[0], 1, 1, 1)).repeat(1, *real.shape[1:]).to(real.device)
    interp = (eps*real+ (1-eps)*fake).to(real.device)
    d_interp = discriminator(interp, None)
    gp = torch.autograd.grad(outputs=d_interp, inputs=interp,
                              grad_outputs=torch.ones_like(d_interp),
                              create_graph=True, retain_graph=True)[0]          
    gp = gp.view(gp.shape[0], -1)
    gp = ((gp.norm(2, dim=1) - 1)**2).mean()     

    d_loss_gp = d_loss + 10*gp
    return d_loss_gp
  
  elif mode == 'g':
    g_loss = -d_fake.mean()   
    return g_loss

def hinge_loss(d_real, d_fake, mode):
  if mode == 'd':
    real_loss = torch.mean(F.relu(1. - d_real))
    fake_loss = torch.mean(F.relu(1. + d_fake))
    d_loss = 0.5*(real_loss + fake_loss)
    return d_loss
  elif mode == 'g':
    g_loss = -d_fake.mean()
    return g_loss

