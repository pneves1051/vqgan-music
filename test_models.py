import platform
import yaml
import os
import sys
import numpy as np
import torch
import torchaudio
if platform.system() == 'Windows':
    torchaudio.set_audio_backend('soundfile')
else:
    torchaudio.set_audio_backend('sox_io')

from datasets.audio_dataset import DummyDataset
from models.vq_vae.vq_vae import VQVAE
from models.discriminator import MultiDiscriminator
from utils.vqvae_trainer import VQVAETrainer
from models.losses import hinge_loss, vqvae_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

os.chdir('VQ_GAN_music')
with open(r'./config/test.yaml') as file:
    hps = yaml.full_load(file)

SAMPLE_RATE = hps['dataset']['sample_rate']
WINDOW_SIZE = int(2**(np.ceil(np.log2(hps['dataset']['win_size']*SAMPLE_RATE)))) # approx 2 seconds
HOP_LEN = int(2**(np.ceil(np.log2(hps['dataset']['hop_len']*SAMPLE_RATE))))
CONT = 4**(len(hps['model']['vqgan']['vqvae']['ch_mult'])-1)

# Dataset creation
dataset = DummyDataset(SAMPLE_RATE, hps['dataset']['win_size'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
real = next(iter(dataloader))['inputs'].to(device)

v_hps = hps['model']['vqgan']['vqvae']
d_hps = hps['model']['vqgan']['disc']
v_num_chs = [v_hps['ch']*mult for mult in v_hps['ch_mult']]
d_num_chs = [d_hps['ch']*mult for mult in d_hps['ch_mult']]

# model creation
vqvae = VQVAE(v_hps['embed_dim'], v_hps['n_embed'], 1, 1, v_num_chs, v_hps['dilation_depth'], v_hps['attn_indices']).to(device)
discriminator = MultiDiscriminator(d_hps['in_ch'], d_num_chs, 3, WINDOW_SIZE, CONT, n_classes=None).to(device)

'''
# Test forward pass
fake, codes = vqvae(real)
print("VQVAE pass done", fake.shape, codes.shape)
d_real = discriminator(real, None)
print("Real Disc pass done", [s.shape for s in d_real])
d_fake = discriminator(real, None)
print("Fake Disc pass done", [s.shape for s in d_fake])

print(d_real, d_fake)

# test loss
# vqvae
v_loss = vqvae_loss(real, fake, codes)
# disc
d_loss = 0
D_x = 0
D_G_z1 = 0
for score_real, score_fake in zip(d_real, d_fake):
    D_x += score_real.mean().item()
    D_G_z1 += score_fake.mean().item()
    #Cálculo do erro no batch de amostras reais
    d_loss += self.gan_loss(score_real, score_fake, mode='d')
'''
gan_trainer = VQVAETrainer(vqvae, discriminator, dataloader, vqvae_loss, hinge_loss, hps, device)
samples = gan_trainer.train(1, 'checkpoint_dir', train_gan=True, log_interval=1)