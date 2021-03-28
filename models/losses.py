import torch
import torch.nn as nn
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

def spectral_loss(output, real, n_fft=1024, hop_length=256):
  output_stft = torch.stft(output, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=output.device), return_complex=True)
  real_stft = torch.stft(real, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=real.device), return_comples = True)
  output_spec = spec(output, n_fft = 1024, hop_length=256)
  real_spec = spec(real, n_fft = 1024, hop_length=256)
  spectral_loss = torch.linalg.norm(output_stft - real_stft, ord=2)
  return spectral_loss

def mel_loss(output, real, n_fft=1024, hop_length=256, sample_rate=44100):
  output_mel,_ = create_mel(output, n_fft, hop_length, sample_rate=sample_rate)
  real_mel,_ = create_mel(real, n_fft, hop_length, sample_rate=sample_rate)
  mel_loss = torch.abs(torch.abs(output_stft)-torch.abs(real_stft))
  return mel_loss

def vq_vae_loss(outputs, targets, codes, beta=0.25):
  return torch.Tensor([mse_loss(outputs, targets), latent_loss(top_codes, beta), latent_loss(bottom_codes, beta)])

# GAN losses

# INCLUDE D OR G TO CALCULATE PREDS AND INTERPOLATION
def wgan_loss(discriminator, d_real, d_fake, mode):
  if mode == 'd':
    d_loss = -(d_real.mean() - d_fake.mean())
    # Gradient penalty
    eps = torch.rand((self.b_size, 1, 1, 1)).repeat(1, *real.shape[1:]).to(self.device)
    interp = (eps*real+ (1-eps)*fake).to(self.device)
    d_interp = discriminator(interp, conditions)
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

