import numpy as np
import scipy
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from models.spec_transforms import spec, squeeze, create_log_mel_spec

def split_spec(specs, window_size, hop_length): 
  B, H, W = specs.shape
  
  #num_specs = 1 + int(np.floor((B - window_size)/hop_length))
  num_specs = int(np.floor(W/window_size))

  split_specs = specs[:, :, :num_specs*window_size]
  
  split_specs = split_specs.reshape(B, H, num_specs, window_size).permute(0,2,1,3)
  split_specs = split_specs.reshape(B*num_specs, H, window_size)

  return split_specs


def create_vgg_features(data, sr_input, params):
  if sr_input != params['sr']:
    data = torchaudio.transforms.Resample(sr_input, params['sr']).to(data.device)(data)
  
  n_fft = int(params['n_fft_sec']*params['sr'])
  hop_length = int(params['hop_length_sec']*params['sr'])
  window_size = int(params['window_size_sec']*params['sr'])
  
  mel_specs = create_log_mel_spec(data, n_fft, hop_length, window_size,
                              params['n_mels'], params['fmin_mel'], 
                              params['fmax_mel'], params['sr'], params['log_eps']) 

  ex_window_size = int(params['ex_window_size_sec']/params['hop_length_sec'])
  ex_hop_length =  int(params['ex_hop_length_sec']/params['hop_length_sec'])
  
  mel_specs = split_spec(mel_specs, ex_window_size, ex_hop_length).unsqueeze(1)

  return mel_specs


def fad_score(classifier, real_features, fake_features):  
  # calculate mean and covariance
  mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
  mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
  # calculate summed square difference
  dif = np.sum((mu1 - mu2)**2.0)
  # calculate sqrt of product between covs
  mean_cov = scipy.linalg.sqrtm(sigma1.dot(sigma2))
  # check for imaginary numbers
  if np.iscomplexobj(mean_cov):
    mean_cov = mean_cov.real
  # calculate FID score
  fad = dif + np.trace(sigma1 + sigma2 - 2.0 * mean_cov)
  return fad


def calculate_fad(generator, classifier, dataloader, device, params, num_samples=10000, sr=44100):#,n_fft=1024, n_mels=128, window_size=1024, hop_length=256):
  real_features_list = []
  fake_features_list = []
  generator.eval()
  classifier.eval()
  i=0
  #mel = torch.Tensor(librosa.filters.mel(44100, n_fft, n_mels=n_mels)).to(device)
  
  #b_size = data = next
  with torch.no_grad():
    #while len(real_features_list*b_size) <= num_samples:
    for data in dataloader:#[:num_samples]: 
      i+=1
      if i%20 ==0 : print(i)
      #data = next(iter(dataloader))
     
      real = data['inputs'].to(device)
                  
      fake = generator(real)

      real = real[:, :, :int(sr*params['sample_size_sec'])]
      fake = fake[:, :, :int(sr*params['sample_size_sec'])]

      real_specs = create_vgg_features(real, sr, params).transpose(-1, -2)
      fake_specs = create_vgg_features(fake, sr, params).transpose(-1, -2)
      
      real_features = classifier(real_specs)
      fake_features = classifier(fake_specs)
            
      real_features = real_features.flatten(1).cpu().numpy()
      fake_features = fake_features.flatten(1).cpu().numpy()
            
      real_features_list.append(real_features)
      fake_features_list.append(fake_features)
      
  real_features_list, fake_features_list = np.concatenate(real_features_list, axis=0), np.concatenate(fake_features_list, axis=0)
  fad = fad_score(classifier, real_features_list, fake_features_list)

  return fad

# assumes images have the shape 299x299x3, pixels in [0,255]
def inception_score(yhat, n_split=10, eps=1E-16):
  # enumerate splits of images/predictions
  scores = list()
  n_part = np.floor(yhat.shape[0] / n_split)
  for i in range(n_split):
    # retrieve p(y|x)
    ix_start, ix_end = int(i * n_part), int(i * n_part + n_part)
    p_yx = yhat[ix_start:ix_end]
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # calculate KL divergence using log probabilities
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the log
    is_score = np.exp(avg_kl_d)
    # store
    scores.append(is_score)
  # average across images
  is_avg, is_std = np.mean(scores), np.std(scores)
  return is_avg, is_std

def calculate_is(generator, classifier, dataloader, device, params, num_samples=10000, sr=44100):
  yhat_list = []
  generator.eval()
  classifier.eval()
  i=0
  with torch.no_grad():
    #while len(real_list*b_size)<=num_samples:
    for data in dataloader: 
      i+=1
      if i%20 ==0 : print(i)          
      
      real = data['inputs'].to(device)
           
      fake = generator(real)
     
      fake = real[:, :, :int(sr*params['sample_size_sec'])]

      fake_specs = create_vgg_features(fake, sr, params).transpose(-1, -2)
      
       # predict class probabilities for images
      yhat = F.softmax(classifier(fake_specs)[1], dim=1).cpu().numpy()
      
      yhat_list.append(yhat)  
  
  yhat_list = np.concatenate(yhat_list, axis=0)
  is_avg, is_std = inception_score(yhat_list)
  return is_avg, is_std
