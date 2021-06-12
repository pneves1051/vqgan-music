import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.spec_transforms import stft, spec, create_mel, squeeze, norm


# VQ_VAE losses

def latent_loss(codes, beta=0.25):
  #z_e, z_q = codes[0], codes[1]
  # Assumes shape= (batch_size, length, emb_dim)
  z_e, z_q = torch.split(codes, codes.shape[-1]//2, dim=-1)
  #print(z_e.norm(), z_q.norm())
  vq_loss=torch.mean((z_e.detach()-z_q)**2)
  commit_loss = torch.mean((z_e-z_q.detach())**2)
  latent_loss = vq_loss+beta*commit_loss
  return latent_loss


def spectral_loss(real, fake, n_fft=1024, hop_length=120, window_size=600, eps=1e-4):
  real_spec = spec(squeeze(real), n_fft = n_fft, hop_length=hop_length, window_size=window_size)
  fake_spec = spec(squeeze(fake), n_fft = n_fft, hop_length=hop_length, window_size=window_size)
  #spectral_loss = torch.linalg.norm(real_spec.view(real_spec.shape[0], -1) 
  #                                  - fake_spec.view(fake_spec.shape[0], -1), ord='fro')
  #spec_loss = norm(real_spec - fake_spec)
  spec_loss = F.mse_loss(fake_spec, real_spec) + F.mse_loss(torch.log(fake_spec + eps), torch.log(real_spec + eps))  
  return spec_loss


def mel_log_spectral_loss(real, fake, n_fft=1024, hop_length=120, window_size=600, n_mels=256, eps=1e-4):
  real_spec = spec(squeeze(real), n_fft = n_fft, hop_length=hop_length, window_size=window_size)# + eps)
  fake_spec = spec(squeeze(fake), n_fft = n_fft, hop_length=hop_length, window_size=window_size)# + eps)
  #mel = torchaudio.functional.create_fb_matrix(n_freqs=n_fft//2+1, n_mels=256, sample_rate=sample_rate, norm=1)
  mel = torch.Tensor(librosa.filters.mel(44100, n_fft, n_mels=n_mels)).to(real.device)
  real_spec = mel@real_spec
  fake_spec = mel@fake_spec
  #spec_loss = norm(real_spec - fake_spec)
  real_spec = torch.clamp(real_spec, min=1e-4)
  fake_spec = torch.clamp(fake_spec, min=1e-4)
  spec_loss = F.mse_loss(torch.log(fake_spec), torch.log(real_spec))# + F.mse_loss(fake_spec, real_spec) 
  return spec_loss


def multispectral_loss(real, fake, n_fft_list=[2048, 1024, 512], hop_length_list=[240, 120, 50], window_size_list = [1200, 600, 240]):
  losses = []
  for n_fft, hop_length, window_size in zip(n_fft_list, hop_length_list, window_size_list):
    losses.append(spectral_loss(real, fake, n_fft, hop_length,  window_size))
    
  return sum(losses) /len(losses)


def mel_multispectral_loss(real, fake, n_fft_list=[2048, 1024, 512], hop_length_list=[240, 120, 50], window_size_list = [1200, 600, 240], n_mels_list=[512, 256, 128]):
  losses = []
  for n_fft, hop_length, window_size, n_mels in zip(n_fft_list, hop_length_list, window_size_list, n_mels_list):
    losses.append(mel_log_spectral_loss(real, fake, n_fft, hop_length, window_size, n_mels))
    
  return sum(losses) /len(losses)


def vqvae_loss(real, fake, codes, beta=0.25, spec_hp=1.0, mel=False):
  l2_loss = F.mse_loss(fake, real)
  lat_loss = latent_loss(codes, beta=beta)
  spec_loss = 0
  if spec_hp != 0.0:
    if mel:
      spec_loss = mel_multispectral_loss(real, fake)#, n_fft_list=[2048], hop_length_list=[240], window_size_list = [1200], n_mels_list=[128])
    else:
      spec_loss = multispectral_loss(real, fake)
  return l2_loss, lat_loss, spec_hp*spec_loss


# GAN losses


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


def hinge_loss(score_real=None, score_fake=None, mode='d'):
  if mode == 'd':
    assert (score_real is not None and score_fake is not None)
    real_loss = torch.mean(F.relu(1. - score_real))
    fake_loss = torch.mean(F.relu(1. + score_fake))
    d_loss = 0.5*(real_loss + fake_loss)
    return d_loss
  elif mode == 'g':
    assert score_fake is not None
    g_loss = -score_fake.mean()
    return g_loss

