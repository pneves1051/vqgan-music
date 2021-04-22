import torch
import torch.nn as nn
import torchaudio

class PitchShift(nn.Module):
  def __init__(self, sample_rate, pitch_factor):
    super(PitchShift, self).__init__()
    self.sample_rate = sample_rate
    self.pitch_factor = pitch_factor

  def __call__(self, data):
    pf = torch.randint(-self.pitch_factor, self.pitch_factor+1, (data.shape[0],))
    data = [torchaudio.sox_effects.apply_effects_tensor(data[i], self.sample_rate, [['pitch', str(int(pf[i])*100)], ['rate', str(self.sample_rate)]])[0] for i in range(len(data))]
    data = torch.stack(data)
    #data = torch.Tensor([librosa.effects.pitch_shift(audio[0], self.sampling_rate, p) for p, audio in zip(pf, data.numpy())])
    #data = data.unsqueeze(1)
    return data

class NoiseInjection(nn.Module):
  def __init__(self, max_svd):
    super(NoiseInjection, self).__init__()
    self.max_svd = max_svd
    
  def forward(self, data):
    noise_svd = self.max_svd*torch.rand(1)[0]
    noise = torch.empty(*(data.shape), device=data.device).normal_(0, noise_svd)
    augmented_data = data + noise
    return augmented_data

class Gain(nn.Module):
  def __init__(self, min_db_gain, max_db_gain):
    super(Gain, self).__init__()
    self.min_db_gain = min_db_gain
    self.max_db_gain = max_db_gain
    
    self.dist = torch.distributions.Uniform(min_db_gain, max_db_gain)

  def db_to_amp_ratio(self, db_gain):
    return torch.pow(10, db_gain/20)

  def forward(self, data):
    augmented_data = torch.mul(data, self.db_to_amp_ratio(self.dist.sample((*data.shape[:-1], 1))).to(data.device))
    return augmented_data

class TimeShift(nn.Module):
  def __init__(self, sampling_rate, shift_min, shift_max):
    super(TimeShift, self).__init__()
    self.sampling_rate = sampling_rate
    self.shift_min = shift_min
    self.shift_max = shift_max
  
  def forward(self, data):
    shifts_list = torch.randint(int(self.sampling_rate*self.shift_min), int(self.sampling_rate*self.shift_max), (data.shape[0],), device=data.device)
    augmented_data = torch.zeros(*data.shape, device=data.device)
    for i in range(len(augmented_data)):
      shift = int(shifts_list[i])
      if shift > 0:
        augmented_data[i][:, shift:] = torch.roll(data[i], shifts=shift, dims=-1)[:, shift:]
      else:
        augmented_data[i][:, :shift] = torch.roll(data[i], shifts=shift, dims=-1)[:, :shift]
    return augmented_data

class PolarityInversion(nn.Module):
  def __init__self(self, p):
    super(PolarityInversion, self).__init__()
    self.p = p
    self.dist = torch.distributions.Uniform(0, 1)
  
  def forward(self, data):
    is_pol =  torch.dist.sample((*data.shape[:-1], 1)) 
    return data

class Transforms(nn.Module):
  def __init__(self, transforms):
    super(Transforms, self).__init__()
    self.transforms = nn.Sequential(*transforms)
  
  def forward(self, data):
    data = self.transforms(data)
    
    return data