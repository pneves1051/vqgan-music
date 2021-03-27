import torch
import torch.nn as nn
from models.spec_transforms import create_mel

def latent_loss(codes, beta=0.25):
  #z_e, z_q = codes[0], codes[1]
  z_e, z_q = torch.split(codes, int(codes.shape[-1]/2), dim=-1)
  #print(z_e.norm(), z_q.norm())
  vq_loss=torch.mean((z_e.detach()-z_q)**2)
  commit_loss = torch.mean((z_e-z_q.detach())**2)
  latent_loss = vq_loss+beta*commit_loss
  return latent_loss

def spectral_loss(output, real):
  output_stft = torch.stft(output)
  real_stft = torch.stft(real)
  spectral_loss = torch.abs(torch.abs(output_stft)-torch.abs(real_stft))
  return spectral_loss

def mel_loss(output, real, n_fft=1024, hop_length=256, sample_rate=44100):
  output_mel,_ = create_mel(output, n_fft, hop_length, sample_rate=sample_rate)
  real_mel,_ = create_mel(real, n_fft, hop_length, sample_rate=sample_rate)
  mel_loss = torch.abs(torch.abs(output_stft)-torch.abs(real_stft))
  return mel_loss

def loss_fn(outputs, targets, top_codes, bottom_codes, beta=0.25):
  return torch.Tensor([mse_loss(outputs, targets), latent_loss(top_codes, beta), latent_loss(bottom_codes, beta)])

