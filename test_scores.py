import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.audio_dataset import DummyDataset
from models.vq_vae.vq_vae import VQVAE
from utils.scores import calculate_fad


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

with open(r'./config/test.yaml') as file:
    hps = yaml.full_load(file)

SAMPLE_RATE = hps['dataset']['sample_rate']
WINDOW_SIZE = int(2**(np.ceil(np.log2(hps['dataset']['win_size']*SAMPLE_RATE)))) # approx 2 seconds
HOP_LEN = int(2**(np.ceil(np.log2(hps['dataset']['hop_len']*SAMPLE_RATE))))
CONT = 4**(len(hps['model']['vqgan']['vqvae']['ch_mult'])-1)

dataset = DummyDataset(SAMPLE_RATE, 1,#hps['dataset']['win_size'],
                       n_iter=10, one_hot=False, mu_law=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

v_hps = hps['model']['vqgan']['vqvae']
d_hps = hps['model']['vqgan']['disc']
v_num_chs = [v_hps['ch']*mult for mult in v_hps['ch_mult']]
d_num_chs = [d_hps['ch']*mult for mult in d_hps['ch_mult']]

# model creation
#vqvae = VQVAE(v_hps['embed_dim'], v_hps['n_embed'], 1, 1, v_num_chs, v_hps['strides'], v_hps['dilation_depth'], v_hps['attn_indices']).to(device)
vqvae = VQVAE(v_hps['embed_dim'], v_hps['n_embed'], 1, 1, v_num_chs, v_hps['strides'],
             v_hps['dilation_depth'], normalization = nn.BatchNorm1d, conv=nn.Conv1d, conv_t = nn.ConvTranspose1d,
             threshold = 1.0).to(device)
vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device, preprocess=False, postprocess=False).to(device)

with open(r'./config/scores.yaml') as file:
   params = yaml.full_load(file)

fad = calculate_fad(vqvae, vggish, dataloader, device, params, num_samples=10000, sr=44100)
