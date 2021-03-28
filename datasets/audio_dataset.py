import random
import os
import pandas as pd
import torchaudio
import torch
import glob
import math

class AudioDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset_dir, annotations_path, sr, window_size, hop_length, batch_size):
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
    annotations_df = pd.read_csv(annotations_path)
    annotations_df.rename(columns = {
          'musicId': 'id',
          'Arousal(mean)': 'arousal',
          'Valence(mean)': 'valence'
          }, inplace=True)
    self.annotations = annotations_df.to_numpy()

    self.sr = sr
    self.window_size = window_size
    self.hop_length = hop_length
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
      annotations=[]
      
      music_id = os.path.basename(str(file)).replace('.mp3', '')
      if music_id != '' and np.any(float(music_id) in self.annotations[:, 0]):
        
        # carrega arquivo de áudio
        signal, orig_sr = torchaudio.load(file)
        signal = torchaudio.transforms.Resample(orig_sr, self.sr)(signal)
        # normalização entre [-1, 1]
        signal = signal - signal.mean()
        signal = signal/signal.abs().max()
        signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        assert not torch.any(signal.abs() > 1.)
        signal = signal[:1, :]

        # anotações de alerta e valência
        annotation = torch.Tensor([a[1:] for a in self.annotations if int(a[0]) == int(music_id)])
              
        for j in range(0, signal.shape[1] -self.window_size, self.hop_length):
          
          current_signal = signal[:, j: j+self.window_size]
          ids.append(torch.Tensor([int(music_id)]))
          audio.append(current_signal)
          #####ATTENTION: NOT USING WHOLE SLICE
          annotations.append(annotation)

          if len(audio) >= batch_size:
            data_batch = {'ids': [], 'inputs': [], 'targets': [], 'annotations': []}
            data_batch['ids'] = torch.stack(ids)
            data_batch['inputs'] =  torch.stack(audio)
            data_batch['targets'] = torch.stack(audio)
            data_batch['annotations'] = torch.stack(annotations)
            
            ids=[]
            audio = []
            annotations = []
            
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
  def __init__(self, dataset_dir, audio_path, sr, window_size, hop_length, batch_size, transform=None):
    """
    Args:
      song_folder (string): Caminho para a pasta com os arquivos.
    """
    self.dataset_dir = dataset_dir
    self.file_list = glob.glob(dataset_dir + audio_path)
    self.classes = [os.path.basename(x[0]) for x in os.walk(dataset_dir)][1:]
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
      music_id = os.path.basename(str(f)).replace('.wav', '')
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
        print(signal.shape)

        # mu scale
        #signal = librosa.mu_compress(signal, quantize=False)[np.newaxis, ...]
        #signal = 2*((torchaudio.transforms.MuLawEncoding(256)(signal) + 1)/256.) -1.
        #signal = torchaudio.transforms.MuLawDecoding()(torchaudio.transforms.MuLawEncoding()(signal))

        #assert not torch.any(signal.abs() > 1.)
        signal = signal.mean(0, keepdim=True)
        
        # condicionando por instrumento
        cond = torch.Tensor(np.zeros(len(self.classes)))
        sig_class = os.path.basename(os.path.dirname(str(f)))
        sig_class_idx = self.classes.index(sig_class)
        cond[sig_class_idx] = 1.
                             
        for j in range(0, signal.shape[1] -self.window_size, self.hop_length):
          current_signal = signal[:, j: j+self.window_size]
                    
          ids.append(torch.Tensor([0]))
          audio.append(current_signal)
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
