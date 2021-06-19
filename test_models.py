import platform
import yaml
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
if platform.system() == 'Windows':
    torchaudio.set_audio_backend('soundfile')
else:
    torchaudio.set_audio_backend('sox_io')

from datasets.audio_dataset import DummyDataset
from models.vq_vae.modules import VectorQuantizer
from models.vq_vae.vq_vae import VQVAE
from models.discriminator import MultiDiscriminator
from utils.vqvae_trainer import VQVAETrainer
from models.losses import hinge_loss, vqvae_loss

from models.transformers.transformer import Transformer
from datasets.transformer_dataset import TransformerDatasetNoCond
from utils.transformer_trainer import TransformerTrainer
from utils.utils import encode_dataset, generate
from utils.augmentations import TimeShift, Gain, Transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

# Test VQ
rand_data = torch.randn(32, 16, 8)
test_vq = VectorQuantizer(embed_dim=8, n_embed=10, decay=0.99, eps=1e-5, threshold=0.0)
for _ in range(10):
    x = test_vq(rand_data)
    #print(test_vq.embed)
    print(test_vq.cluster_size)


#os.chdir('VQ_GAN_music')
with open(r'./config/test.yaml') as file:
    hps = yaml.full_load(file)

SAMPLE_RATE = hps['dataset']['sample_rate']
WINDOW_SIZE = int(2**(np.ceil(np.log2(hps['dataset']['win_size']*SAMPLE_RATE)))) # approx 2 seconds
HOP_LEN = int(2**(np.ceil(np.log2(hps['dataset']['hop_len']*SAMPLE_RATE))))
CONT = 4**(len(hps['model']['vqgan']['vqvae']['ch_mult'])-1)

# Dataset creation
dataset = DummyDataset(SAMPLE_RATE, hps['dataset']['win_size'], one_hot=False, mu_law=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
real = next(iter(dataloader))['inputs'].to(device)

# augmentations
transforms = Transforms((Gain(-18, +6), TimeShift(SAMPLE_RATE, -0.25, 0.25)))

v_hps = hps['model']['vqgan']['vqvae']
d_hps = hps['model']['vqgan']['disc']
v_num_chs = [v_hps['ch']*mult for mult in v_hps['ch_mult']]
d_num_chs = [d_hps['ch']*mult for mult in d_hps['ch_mult']]


# model creation
vqvae = VQVAE(v_hps['embed_dim'], v_hps['n_embed'], 1, 1, v_num_chs, v_hps['strides'], v_hps['dilation_depth'], v_hps['attn_indices']).to(device)
discriminator = MultiDiscriminator(d_hps['in_ch'], d_num_chs, d_hps['stride'], 3, WINDOW_SIZE, CONT, n_classes=None).to(device)

gan_trainer = VQVAETrainer(vqvae, discriminator, dataloader, vqvae_loss, hinge_loss, hps, device, transforms=transforms)
samples = gan_trainer.train(1, 'checkpoint_dir', train_gan=True, log_interval=1)

tr_seq_len = (WINDOW_SIZE//2)//CONT
print(tr_seq_len)
tr_data = encode_dataset(dataloader, vqvae, device)
tr_dataset = TransformerDatasetNoCond(tr_data, tr_seq_len)
print(tr_dataset.dataset)
tr_dataloader = dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=hps['dataset']['tr_b_size'])
tr_data = next(iter(tr_dataloader))
print(tr_data)

'''
tr_hps = hps['model']['transformer']
tr_vocab_size = tr_hps['vocab_size']
tr_d_model = tr_hps['d_model']
tr_n_head = tr_hps['n_head']
tr_n_layer = tr_hps['n_layer']
tr_max_len = tr_hps['max_len']
codebook = vqvae.get_vqvae_codebook()
tr_lr = float(tr_hps['lr'])

transformer = Transformer(tr_vocab_size, tr_d_model, tr_n_head, tr_n_layer, tr_max_len, codebook).to(device)
tr_loss_fn = nn.CrossEntropyLoss()

transformer_trainer = TransformerTrainer(transformer, tr_dataloader, None, tr_loss_fn, device, tr_lr)
transformer_trainer.train(10, 'checkpoint_dir', log_interval=1)

generated = generate(real, None, vqvae, transformer, WINDOW_SIZE, WINDOW_SIZE, CONT, 1, device)
print(generated.shape)
'''