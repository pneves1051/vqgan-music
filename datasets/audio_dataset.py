import random
import traceback
import os
import pandas as pd
import librosa
import glob
import math
import numpy as np
import torchaudio
import torch
import torch.nn.functional as F


class AudioDatasetNoCond(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, sr, window_size, hop_len, batch_size, shuffle=True, use_torch=True, extension='.wav', one_hot=False, mu_law=False):
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
    self.shuffle = shuffle
    self.use_torch = use_torch
    self.extension = extension
    self.one_hot = one_hot
    self.mu_law = mu_law

    self.start = 0
    self.end = len(self.file_list)
        
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list

  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    if self.shuffle:
      random.shuffle(shuffled_list)    
    
    audio=[]
    ids = []
    for file in shuffled_list:
      id, ext = os.path.splitext(os.path.basename(str(file)))
      if id != '' and ext == self.extension:
        # load file
        if self.use_torch:
          signal, orig_sr = torchaudio.load(file)
          #signal = signal.type(torch.float16)
        else:
          signal, orig_sr = librosa.load(file, sr=self.sr, mono=False)
          signal = torch.Tensor(signal)
        if self.sr != orig_sr:
          signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
               
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()        
        # normalization
        if self.mu_law:
          signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        
        assert not torch.any(signal.abs() > 1.)
        signal = signal.mean(0, keepdim=True)
        if self.one_hot:
          signal = torchaudio.transforms.MuLawEncoding(256)(signal)
          signal = F.one_hot(signal)[0].transpose(-1,-2)
        
        sliced_signal = torch.stack(torch.split(signal, self.window_size, dim=-1)[:-1])
        if self.shuffle: 
          sliced_signal = sliced_signal[torch.randperm(len(sliced_signal))]

        for s in sliced_signal:
          try:
            ids.append(torch.Tensor([int(id)]))
          except:
            ids.append(torch.Tensor([self.file_list.index(file)]))
          audio.append(s)
          if len(audio) >= self.batch_size:
            if self.shuffle:
              shuffled_inputs = list(zip(ids, audio))
              random.shuffle(shuffled_inputs)
              ids, audio = zip(*shuffled_inputs)

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


class AudioDatasetCond(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, sr, window_size, hop_len, batch_size, shuffle=True, use_torch=True, extension='.wav', one_hot=False, mu_law=False, csv_path=None, downmix_augment=False, split='train'):
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
    #print(self.file_list)
    
    self.sr = sr
    self.window_size = int(2**(np.ceil(np.log2(sr*window_size))))
    self.hop_len = int(2**(np.ceil(np.log2(sr*hop_len))))
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.use_torch = use_torch
    self.extension = extension
    self.one_hot = one_hot
    self.mu_law = mu_law

    if csv_path is not None:
      self.csv = pd.read_csv(csv_path)#, delimiter='\t')
      self.csv = self.csv.set_index('audio_filename')

      self.classes = pd.unique(self.csv['canonical_composer']).tolist()
          
    else:
      self.csv = None

    self.downmix_augment = downmix_augment

    self.split = split

    self.start = 0
    self.end = len(self.file_list)
        
  @property
  def shuffled_file_list(self):
    shuffled_list = self.file_list.copy()
    random.shuffle(shuffled_list)
    return shuffled_list
  
  def open_file(self, file):
    # load file
    if self.use_torch:
      signal, orig_sr = torchaudio.load(file)
      #signal = signal.type(torch.float16)
    else:
      signal, orig_sr = librosa.load(file, sr=self.sr, mono=False)
      signal = torch.Tensor(signal)
    if self.sr != orig_sr:
      signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
    
    return signal
  
  def downmix(self, signal):
    if signal.shape[0] == 2 and self.downmix_augment:
      d = torch.rand(1).item()
      signal = (d*signal[0] + (1-d)*signal[1]).unsqueeze(0)
    else: 
      signal = signal.mean(0, keepdim=True)
    return signal

  def process_signal(self, signal):
     # normalization
    signal = signal - signal.mean()
    signal = signal/signal.abs().max()   
    if self.mu_law:
      signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.

    assert not torch.any(signal.abs() > 1.)

    if self.one_hot:
      signal = torchaudio.transforms.MuLawEncoding(256)(signal)
      signal = F.one_hot(signal)[0].transpose(-1,-2)
    
    return signal

  def slice_signal(self, signal):
    split_signal = torch.split(signal, self.window_size, dim=-1)
    split_signal = split_signal[:signal.shape[-1]//self.window_size]
    #if split_signal[-1].shape[-1] < self.window_size:
    #  split_signal = split_signal[:-1]
    sliced_signal = torch.stack(split_signal)
    if self.shuffle: 
      sliced_signal = sliced_signal[torch.randperm(len(sliced_signal))]
    return sliced_signal
  
  def create_batch(self, audio, ids, conditions):
    if self.shuffle:
      shuffled_inputs = list(zip(ids, audio, conditions))
      random.shuffle(shuffled_inputs)
      ids, audio, conditions = zip(*shuffled_inputs)
    
    batch = {'ids': [], 'inputs': [], 'conditions': []}
    batch['ids'] = torch.stack(ids)
    batch['inputs'] =  torch.stack(audio)
    batch['conditions'] = torch.stack(conditions)

    return batch

  def read_dataset(self, file_list, batch_size):
    shuffled_list = file_list.copy()
    if self.shuffle:
      random.shuffle(shuffled_list)    
    
    audio=[]
    ids = []
    conds = []
    for file in shuffled_list:
      try:
        id, ext = os.path.splitext(os.path.basename(str(file)))
        folder = os.path.basename(os.path.dirname(file))
        if id != '' and ext == self.extension and self.split == self.csv.loc[folder + '/' + id + self.extension, 'split']:
          signal = self.open_file(file)
            
          cond = self.classes.index(self.csv.loc[folder + '/' + id + self.extension, 'canonical_composer'])
          
          if torch.any(signal != 0.):
            assert len(signal.shape) == 2
            signal = self.downmix(signal)

            signal = self.process_signal(signal)
            sliced_signal = self.slice_signal(signal)
            for s in sliced_signal:
              try:
                ids.append(torch.Tensor([int(id)]))
              except:
                ids.append(torch.Tensor([self.file_list.index(file)]))
              audio.append(s)
              conds.append(torch.Tensor([int(cond)]))

              if len(audio) >= self.batch_size:
                batch = self.create_batch(audio, ids, conds)
                ids = []
                audio = []
                conds = []
                yield batch      

      except RuntimeError as error:
        traceback.print_exc()

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
    self.window_size = int(2**(np.ceil(np.log2(sr*window_size))))
    self.hop_len = int(2**(np.ceil(np.log2(sr*hop_len))))
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
        
        # loads audio file
        signal, orig_sr = torchaudio.load(file)
        if self.sr != orig_sr:
          signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        # normalization between [-1, 1]
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
  def __init__(self, sr, window_size, n_iter=1, one_hot=False, mu_law=False):
    self.sr = sr
    self.window_size = int(2**(np.ceil(np.log2(sr*window_size))))
    self.n_iter=n_iter
    self.one_hot = one_hot
    self.mu_law = mu_law

  def produce_random_batch(self):
    for _ in range(self.n_iter):
      signal = torch.randn((1, 1, self.window_size))
      signal = signal - signal.mean()
      signal = signal/signal.abs().max()
      if self.mu_law:
          signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
      if self.one_hot:
          signal = torchaudio.transforms.MuLawEncoding(256)(signal)
          signal = F.one_hot(signal)[0].transpose(-1, -2)
          print(signal.shape)
      batch = {'ids': torch.LongTensor([0]), 'inputs': signal, 'conditions': None}
      yield batch
    
  def __iter__(self):
    return self.produce_random_batch()
