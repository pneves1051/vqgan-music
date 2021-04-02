import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
torchaudio.set_audio_backend("soundfile")


def inst_freq_np(phase):
  un_phase = np.unwrap(phase, axis=1)
  un_phase_diff = np.diff(un_phase, axis=1)
  i_freq = np.concatenate([phase[:, :1], un_phase_diff], axis=1)

  return i_freq/np.pi

def inv_inst_freq_np(i_freq):
  i_freq_inv = np.cumsum(i_freq * np.pi, axis=1)
  return i_freq_inv

def exp_s_np(x):
  return np.exp(x) - 1e-10

def log_s_np(x):
  return np.log(x + 1e-10)

def stft_log_np(x):
  mag = x[0]
  phase = x[1]
  mlog = log_s(mag)
  return np.stack([mlog, phase])

def stft_exp_np(x):
  mag = x[0]
  phase = x[1]
  mexp = exp_s(mag)
  return np.stack([mexp, phase])

def normalize_np(x, min, max, clip = False):
  norm_x = x
  if clip:
    norm_x = np.clip(norm_x, min, max)
  norm_x = 2*(norm_x - min)/(max-min) - 1
  return norm_x

def denormalize_np(x, min, max):
  norm_x = (x+1)*(max-min)/2 + min
  return norm_x

def create_stft_np(signal, n_fft, hop_length):
  stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True)
  #stft= np.pad(stft, [[0, 0], [1, 1 2]], mode = reflect)
  mag, phase = librosa.magphase(stft)
  phase = np.angle(phase)
  
  mag = np.abs(mag)
  #mag = mag - mag.mean()
  #mag = mag/np.max(np.abs(mag))
  #mag = (mag - np.min(mag))/(np.max(mag)-np.min(mag))

  log_mag = log_s(mag)
  inst_f = inst_freq(phase)

  #print(log_mag.min(), log_mag.max())
  #print(inst_f.min(), inst_f.max())

  log_mag = normalize(log_mag, -12, 6, clip=True)
  inst_f = normalize(inst_f, -1, 1, clip=True)
  #print(log_mag.shape)

  spec = np.stack([log_mag, inst_f])

  spec = spec[:, :-1, :-1]
  return spec

def invert_stft_np(spec, hop_length):
  spec = np.pad(spec, ((0, 0), (0, 1) , (0, 1)), mode='mean')
  log_mag, inst_f = spec[0], spec[1]
  #print(log_mag.shape)
  
  log_mag = denormalize(log_mag, -12, 6)
  inst_f = denormalize(inst_f, -1, 1)  
  
  mag = exp_s(log_mag)
  phase = inv_inst_freq(inst_f)

  stft = mag*np.exp(1j * phase)
 
  inv_stft = librosa.istft(stft, hop_length=hop_length)
  return inv_stft, stft

def create_mel_np(signal, n_fft, hop_length):
  stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True)
  #stft= np.pad(stft, [[0, 0], [1, 1 2]], mode = reflect)
  #mag, phase = librosa.magphase(stft)

  mag = np.abs(stft)
  phase = np.angle(stft)
  
  #mag = mag - mag.mean()
  #mag = mag/np.max(np.abs(mag))
  #mag = (mag - np.min(mag))/(np.max(mag)-np.min(mag))
  
  mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=256, norm=1)
  # mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft = n_fft, n_mels=n_fft//2+1, norm=1)
  #print(mel.shape, stft.shape)

  mel_mag = mel@mag
  mel_phase = mel@phase

  log_mel_mag = log_s(mel_mag)
  mel_if = inst_freq(mel_phase)

  #print(log_mel_mag.min(), log_mel_mag.max())
  #print(mel_if.min(), mel_if.max())

  log_mel_mag = normalize(log_mel_mag, -10, 6, clip=True)
  mel_if = normalize(mel_if, -1, 1, clip=True)
  #print(log_mel_mag.shape)

  mel_spec = np.stack([log_mel_mag, mel_if])

  mel_spec = mel_spec[:, :, :-1]
  #print(mel_spec.shape)
  return mel_spec

def invert_mel_np(mel_spec, n_fft, hop_length):
  mel_spec = np.pad(mel_spec, ((0, 0), (0, 0) , (0, 1)), mode='reflect')
  
  log_mel_mag, mel_if = mel_spec[0], mel_spec[1]
  
  log_mel_mag = denormalize(log_mel_mag, -10, 6)
  mel_if = denormalize(mel_if, -1, 1)
  
  
  mel_mag = exp_s(log_mel_mag)
  mel_phase = inv_inst_freq(mel_if)
  
  # mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_fft//2+1)
  mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=256, norm=1)

  inv_mel = np.linalg.pinv(mel)

  mag = inv_mel@mel_mag
  phase = inv_mel@mel_phase

  stft = mag*np.exp(1j * phase)
 
  inv_stft = librosa.istft(stft, hop_length=hop_length)
  return inv_stft, stft

### TORCH ###

pi = torch.acos(torch.zeros(1)).item()

def stft(input, n_fft=1024, hop_length=256):
  return torch.stft(output, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=output.device), return_complex=True)
  
def spec(stft):
  return torch.linalg.norm(stft, ord='fro', dim=-1)

def diff(inputs, dim=-1):
  size = inputs.shape
  slice_back = torch.narrow(inputs, dim, 0, size[dim]-1)
  slice_front = torch.narrow(inputs, dim, 1, size[dim]-1) 
  diffs = slice_front - slice_back
  return diffs

def unwrap(inputs, dim=-1):
  diffs = diff(phases, dim=dim)
  mods = torch.fmod(diffs + pi, pi * 2.0) - pi
  indices = torch.logical_and(torch.equal(mods, -pi), torch.greater(diffs, 0.0))
  mods = torch.where(indices, torch.ones_like(mods) * pi, mods)
  corrects = mods - diffs
  cumsums = torch.cumsum(corrects, dim = dim)
  shape = phases.shape
  shape[axis] = 1
  cumsums = torch.cat([tf.zeros(shape, device=inputs.device), cumsums], dim=dim)
  unwrapped = phases + cumsums
  return unwrapped                         

def inst_freq(phase):
  un_phase = unwrap(phase, axis=1)
  un_phase_diff = diff(un_phase, axis=1)
  i_freq = torch.cat([phase[:, :1], un_phase_diff], dim=1)

  return i_freq/np.pi

def inv_inst_freq(i_freq):
  i_freq_inv = torch.cumsum(i_freq * np.pi, dim=1)
  return i_freq_inv

def exp_s(x):
  return torch.exp(x) - 1e-10

def log_s(x):
  return torch.log(x + 1e-10)

def stft_log(x):
  mag = x[0]
  phase = x[1]
  mlog = log_s(mag)
  return torch.stack([mlog, phase], dim=1)

def stft_exp(x):
  mag = x[0]
  phase = x[1]
  mexp = exp_s(mag)
  return torch.stack([mexp, phase])

def normalize(x, min, max, clip = False):
  norm_x = x
  if clip:
    norm_x = torch.clip(norm_x, min, max)
  norm_x = 2*(norm_x - min)/(max-min) - 1
  return norm_x

def denormalize(x, min, max):
  norm_x = (x+1)*(max-min)/2 + min
  return norm_x

def create_stft(signal, n_fft, hop_length):
  stft = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True, return_comples=True)
 
  mag, phase = torchaudio.functional.magphase(stft)
  
  phase = np.angle(phase)
  mag = torch.abs(mag)
  #mag = mag - mag.mean()
  #mag = mag/np.max(np.abs(mag))
  #mag = (mag - np.min(mag))/(np.max(mag)-np.min(mag))

  log_mag = log_s(mag)
  inst_f = inst_freq(phase)

  #print(log_mag.min(), log_mag.max())
  #print(inst_f.min(), inst_f.max())

  log_mag = normalize(log_mag, -12, 6, clip=True)
  inst_f = normalize(inst_f, -1, 1, clip=True)
  #print(log_mag.shape)

  spec = torch.stack([log_mag, inst_f], dim=1)

  spec = spec[:, :, :-1, :-1]
  return spec

def invert_stft(spec, hop_length):
  spec = F.pad(spec, (0, 1, 0, 1), mode='mean')
  log_mag, inst_f = spec[:, 0], spec[:, 1]
  #print(log_mag.shape)
  
  log_mag = denormalize(log_mag, -12, 6)
  inst_f = denormalize(inst_f, -1, 1)  
  
  mag = exp_s(log_mag)
  phase = inv_inst_freq(inst_f)

  stft = mag*torch.exp(1j * phase)
 
  inv_stft = torch.istft(stft, hop_length=hop_length)
  return inv_stft, stft

def create_mel(signal, n_fft, hop_length, sample_rate=44100):
  stft = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True)
  #stft= np.pad(stft, [[0, 0], [1, 1 2]], mode = reflect)
  #mag, phase = librosa.magphase(stft)

  mag = torch.abs(stft)
  phase = torch.angle(stft)
  
  #mag = mag - mag.mean()
  #mag = mag/np.max(np.abs(mag))
  #mag = (mag - np.min(mag))/(np.max(mag)-np.min(mag))
  
  mel = torchaudio.functional.create_fb_matrix(n_freqs=n_fft//2+1, n_mels=256, sample_rate=sample_rate, norm=1)
  mel = mel.unsqueeze(0)
  #print(mel.shape, stft.shape)

  mel_mag = torch.matmul(mel, mag)
  mel_phase = torch.matmul(mel, phase)

  log_mel_mag = log_s(mel_mag)
  mel_if = inst_freq(mel_phase)

  #print(log_mel_mag.min(), log_mel_mag.max())
  #print(mel_if.min(), mel_if.max())

  log_mel_mag = normalize(log_mel_mag, -10, 6, clip=True)
  mel_if = normalize(mel_if, -1, 1, clip=True)
  #print(log_mel_mag.shape)

  mel_spec = torch.stack([log_mel_mag, mel_if], dim=1)

  mel_spec = mel_spec[:, :, :, :-1]
  #print(mel_spec.shape)
  return mel_spec

def invert_mel(mel_spec, n_fft, hop_length, sample_rate=44100):
  mel_spec = torch.pad(mel_spec, (0, 1), mode='reflect')
  
  log_mel_mag, mel_if = mel_spec[:, 0], mel_spec[:, 1]
  
  log_mel_mag = denormalize(log_mel_mag, -10, 6)
  mel_if = denormalize(mel_if, -1, 1)
  
  mel_mag = exp_s(log_mel_mag)
  mel_phase = inv_inst_freq(mel_if)
  
  # mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_fft//2+1)
  mel = torchaudio.functional.create_fb_matrix(n_freqs=n_fft//2+1, n_mels=256, sample_rate=sample_rate, norm=1)

  inv_mel = torch.linalg.pinv(mel).unsqueeze(0)

  mag = torch.matmul(inv_mel, mel_mag)
  phase = torch.matmul(inv_mel, mel_phase)

  stft = mag*torch.exp(1j * phase)
 
  inv_stft = torch.istft(stft, hop_length=hop_length)
  return inv_stft, stft
