import random
import os
import pandas as pd
import librosa
import torchaudio
import torch
import glob
import math
import numpy as np

class AudioDatasetNoCond(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, sr, window_size, hop_len, batch_size, use_torch=True, extension='.wav'):
    """
    Args:
      dataset_dir (string): dataset directory
      sr: sample rate of the audio
      window_size: number of samples of each item
      hop_length: step size
      batch_size: batch_size
      use_torch: whether to use torchaudio or not
    """
    self.dataset_dir = dataset_dir
    self.file_list = glob.glob(dataset_dir)
    
    self.sr = sr
    self.window_size = int(2**(np.ceil(np.log2(sr*window_size))))
    self.hop_len = int(2**(np.ceil(np.log2(sr*hop_len))))
    self.batch_size = batch_size
    self.use_torch = use_torch
    self.extension = extension

    self.start = 0
    self.end = len(self.file_list)
        
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    random.shuffle(shuffled_list)    
    
    audio=[]
    ids = []
    for file in shuffled_list:
      id, ext = os.path.splitext(os.path.basename(str(file)))
      if id != '' and ext == self.extension:
        # load file
        if self.use_torch:
          signal, orig_sr = torchaudio.load(file)
        else:
          signal, orig_sr = librosa.load(file, sr=self.sr, mono=False)
          signal = torch.Tensor(signal)
        if self.sr != orig_sr:
          signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        # normalization
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()
        #signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        assert not torch.any(signal.abs() > 1.)
        signal = signal.mean(0, keepdim=True)
        for j in range(0, signal.shape[1] - self.window_size + 1, self.hop_len):
          
          current_signal = signal[:, j: j+self.window_size]
          try:
            ids.append(torch.Tensor([int(id)]))
          except:
            ids.append(torch.Tensor([self.file_list.index(file)]))
          audio.append(current_signal)
          if len(audio) >= self.batch_size:
            batch = {'ids': [], 'inputs': [], 'conditions': None}
            batch['ids'] = torch.stack(ids)
            batch['inputs'] =  torch.stack(audio)
                      
            ids = []
            audio = []
                       
            yield batch        

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = self.start
      iter_end = self.end
    else:
      # divide carga de trabalho
      per_worker = int(math.ceil((self.end-self.start)/ float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return self.read_dataset(self.file_list[iter_start: iter_end],self.batch_size)
    #return self.get_streams()
    #return self.get_batch(self.shuffled_file_list)
    #return self.read_dataset(self.shuffled_file_list, self.batch_size)

class AudioDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, sr, window_size, hop_len, batch_size):
    """
    Args:
      dataset_dir (string): dataset directory
      sr: sample rate of the audio
      window_size: number of samples of each item
      hop_length: step size
      batch_size: batch_size
    """
    self.dataset_dir = dataset_dir
    self.file_list = glob.glob(dataset_dir)
    '''
    conditions_df = pd.read_csv(conditions_path)
    conditions_df.rename(columns = {
          'musicId': 'id',
          'Arousal(mean)': 'arousal',
          'Valence(mean)': 'valence'
          }, inplace=True)
    self.conditions = conditions_df.to_numpy()
    '''
    self.sr = sr
    self.window_size = int(2**(np.ceil(np.log2(SAMPLE_RATE*window_size))))
    self.hop_len = int(2**(np.ceil(np.log2(SAMPLE_RATE*hop_len))))
    self.batch_size = batch_size

    self.start = 0
    self.end = len(self.file_list)
    
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    random.shuffle(shuffled_list)    
    for file in shuffled_list:
      ids=[]
      audio=[]
      conditions=[]
      
      music_id = os.path.basename(str(file)).replace('.mp3', '')
      if music_id != '' and np.any(float(music_id) in self.conditions[:, 0]):
        
        # carrega arquivo de áudio
        signal, orig_sr = torchaudio.load(file)
        if self.sr != orig_sr:
          signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        # normalização entre [-1, 1]
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()
        #signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        assert not torch.any(signal.abs() > 1.)
        signal = signal.mean(0, keepdim=True)

        # anotações de alerta e valência
        condition = torch.Tensor([a[1:] for a in self.conditions if int(a[0]) == int(music_id)])
              
        for j in range(0, signal.shape[1] -self.window_size + 1 , self.hop_length):
          
          current_signal = signal[:, j: j+self.window_size]
          ids.append(torch.Tensor([int(music_id)]))
          audio.append(current_signal)
          #####ATTENTION: NOT USING WHOLE SLICE
          conditions.append(condition)

          if len(audio) >= batch_size:
            data_batch = {'ids': [], 'inputs': [], 'targets': [], 'conditions': []}
            data_batch['ids'] = torch.stack(ids)
            data_batch['inputs'] =  torch.stack(audio)
            data_batch['targets'] = torch.stack(audio)
            data_batch['conditions'] = torch.stack(conditions)
            
            ids=[]
            audio = []
            conditions = []
            
            yield data_batch        

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = self.start
      iter_end = self.end
    else:
      # divide carga de trabalho
      per_worker = int(math.ceil((self.end-self.start)/ float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return self.read_dataset(self.file_list[iter_start: iter_end],self.batch_size)
    #return self.get_streams()
    #return self.get_batch(self.shuffled_file_list)
    #return self.read_dataset(self.shuffled_file_list, self.batch_size)

    
class AudioDataset2(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, audio_path, metadata_path, sr, window_size, hop_length, batch_size, transform=None):
    """
    Args:
      song_folder (string): Caminho para a pasta com os arquivos.
    """
    self.dataset_dir = dataset_dir
    self.file_list = glob.glob(dataset_dir + audio_path)
    self.metadata = pd.read_csv(metadata_path).set_index('uuid4')
    self.classes = self.metadata['instrument'].unique().tolist()
    print(self.classes)

    self.sr = sr
    self.window_size = window_size
    self.hop_length = hop_length
    self.batch_size = batch_size
    self.transform = transform

    self.start = 0
    self.end = len(self.file_list)
    
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list
 
  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    random.shuffle(shuffled_list)    
    ids=[]
    audio = []
    conditions = []
    for f in shuffled_list:
      music_id = os.path.basename(str(f)).replace('.wav', '').split('_')[-1]
      if music_id != '':
                
        # carrega arquivo de áudio
        #samplerate, signal = wavfile.read(f)
        #signal = np.transpose(signal, axes=(0,1))
        #signal = torch.Tensor(signal).transpose(0,1)
        signal, orig_sr = torchaudio.load(f)
        #signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        '''
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          signal, _ = librosa.load(file, sr=self.sr)
        '''
        # normalização entre [-1, 1]
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()
        
        # mu scale
        #signal = librosa.mu_compress(signal, quantize=False)[np.newaxis, ...]
        #signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        #signal = torchaudio.transforms.MuLawDecoding()(torchaudio.transforms.MuLawEncoding()(signal))

        #assert not torch.any(signal.abs() > 1.)
        signal = signal.mean(0, keepdim=True)
        
        
        # condicionando por instrumento
        cond = torch.Tensor(np.zeros(len(self.classes)))
        sig_class = self.metadata.loc[music_id, 'instrument']
        sig_class_idx = self.classes.index(sig_class)
        cond[sig_class_idx] = 1.
                             
        ids.append(torch.Tensor([0]))
        audio.append(signal)
        conditions.append(cond)

        if len(audio) >= batch_size:
          #all_data = list(zip(ids, audio, conditions))
          #random.shuffle(all_data)
          #ids, audio, conditions = zip(*all_data)
          
          data_batch = {'ids': [], 'inputs': [], 'conditions': []}
          data_batch['ids'] = torch.stack(ids)
          data_batch['inputs'] = torch.stack(audio)
          data_batch['conditions'] = torch.stack(conditions)
                      
          ids=[]
          audio = []
          conditions = []
          
          if self.transform:
            data_batch = self.transform(data_batch)

          yield data_batch        
 
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      iter_start = self.start
      iter_end = self.end
    else:
      # divide carga de trabalho
      per_worker = int(math.ceil((self.end-self.start)/ float(worker_info.num_workers)))
      worker_id = worker_info.id
      iter_start = self.start + worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.end)
    return self.read_dataset(self.file_list[iter_start: iter_end],self.batch_size)

class DummyDataset(torch.torch.utils.data.IterableDataset):
  def __init__(self, sr, window_size):
    self.sr = sr
    self.window_size = int(2**(np.ceil(np.log2(sr*window_size))))

  def produce_random_batch(self, n_iter=1):
    for _ in range(n_iter):
      batch = {'inputs': torch.randn((1, 1, self.window_size)), 'conditions': None}
      yield batch
    
  def __iter__(self):
    return self.produce_random_batch(1)
