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

#os.chdir('VQ_GAN_music')
with open(r'./config/fma.yaml') as file:
    hps = yaml.full_load(file)

SAMPLE_RATE = hps['dataset']['sample_rate']
WINDOW_SIZE = int(2**(np.ceil(np.log2(hps['dataset']['win_size']*SAMPLE_RATE)))) # approx 2 seconds
HOP_LEN = int(2**(np.ceil(np.log2(hps['dataset']['hop_len']*SAMPLE_RATE))))
CONT = 4**(len(hps['model']['vqgan']['vqvae']['ch_mult'])-1)

# Dataset creation
dataset = DummyDataset(SAMPLE_RATE, 2)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
random_input = next(iter(dataloader))['inputs'].to(device)

v_hps = hps['model']['vqgan']['vqvae']
d_hps = hps['model']['vqgan']['disc']
v_num_chs = [v_hps['ch']*mult for mult in v_hps['ch_mult']]
d_num_chs = [d_hps['ch']*mult for mult in d_hps['ch_mult']]

# model creation
vqvae = VQVAE(v_hps['embed_dim'], v_hps['n_embed'], 1, 1, v_num_chs, v_hps['dilation_depth'], v_hps['attn_indices']).to(device)
discriminator = MultiDiscriminator(d_hps['in_ch'], d_num_chs, 3, WINDOW_SIZE, CONT, n_classes=None).to(device)

# Test forward pass
reconstructed, codes = vqvae(random_input)
print("VQVAE pass done", reconstructed.shape, codes.shape)
d_real = discriminator(random_input, None)
print("Real Disc pass done", [s.shape for s in d_real])
d_fake = discriminator(random_input, None)
print("Fake Disc pass done", [s.shape for s in d_fake])

print(d_real, d_fake)

gan_trainer = VQVAETrainer(vqvae, discriminator, music_dataloader, vqvae_loss, hinge_loss, hps, device)
samples = gan_trainer.train(1, 'checkpoint_dir', train_gan=True)