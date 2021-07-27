import torch
import torch.nn.functional as F
import time
import numpy as np
from collections import defaultdict
import IPython.display as ipd


class VQVAETrainer():
  def __init__(self, vqvae, discriminator, dataloader, vqvae_loss, gan_loss, hps, device, val_dataloader = None, transforms=None, mel=True):
    self.vqvae = vqvae
    self.discriminator = discriminator    
    self.dataloader = dataloader
    self.vqvae_loss = vqvae_loss
    self.gan_loss = gan_loss
    self.device=device
    self.val_dataloader = val_dataloader
    self.transforms=transforms
    self.mel=mel
    
    self.spec_hp = hps['model']['vqgan']['vqvae']['loss']['spectral_hp']
    self.feat_hp = hps['model']['vqgan']['vqvae']['loss']['feat_hp']
    self.disc_steps = hps['model']['vqgan']['disc']['disc_steps']
 
    betas=(0.5, 0.9)
    self.v_optimizer = torch.optim.Adam(self.vqvae.parameters(), 
                                        lr = float(hps['model']['vqgan']['vqvae']['lr']),
                                        betas=betas)
    self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(),
                                        lr = float(hps['model']['vqgan']['disc']['lr']),
                                        betas=betas)
    #self.e_scheduler = torch.optim.lr_scheduler.StepLR(self.e_optimizer, 1.0, gamma=0.95) 
    #self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, 1.0, gamma=0.95) 
    #self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, 1.0, gamma=0.95) 
 
    self.hps=hps
 
    data = next(iter(self.dataloader))
    self.fixed_real = data['inputs'].to(self.device)
    self.fixed_conditions = data['conditions']
    if self.fixed_conditions is not None:
      self.fixed_conditions = self.fixed_conditions.to(self.device)
    
  def get_loss_hp(self, rec_loss, g_loss, last_layer):
    rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    
    disc_weight = torch.linalg.norm(rec_grads.flatten(1), 'fro') /(torch.linalg.norm(g_grads.flatten(1), 'fro') + 1e-4)
    disc_weight = torch.clamp(disc_weight, 0.0, 1e5).detach()
    
    return disc_weight  
  
  def train_epoch(self, log_interval=20):
    self.vqvae.train()
           
    vqvae_losses = []
    start_time = time.time()
    index=0
    for data in self.dataloader:
      real = data['inputs'].to(self.device)
      conditions = data['conditions']
      if conditions is not None:
        conditions = conditions.to(self.device)
      
      fake, codes = self.vqvae(real)
      
      # loss, loss_list = self.loss_fn(outputs, targets, top_codes, bottom_codes)
      l2_loss, lat_loss, spec_loss = self.vqvae_loss(real, fake, codes, spec_hp=self.spec_hp, mel=self.mel)
      total_loss = l2_loss + lat_loss + spec_loss
      
      self.v_optimizer.zero_grad()
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 1.0)     
      self.v_optimizer.step()
      
      index += 1        
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        print('| {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
              'losses {} |'.format(
              index, 2, #self.g_scheduler.get_last_lr()[0],
              elapsed*1000/log_interval,
              np.mean(vqvae_losses, axis=0)))
        start_time = time.time()
 
      vqvae_losses.append([l2_loss.item(), lat_loss.item(), spec_loss.item() if type(spec_loss) == torch.Tensor else spec_loss])
     
    return np.mean(vqvae_losses, 0), vqvae_losses
    
  def train_epoch_gan(self, log_interval=20):
    self.vqvae.train()
    self.discriminator.train()
 
    start_time = time.time()
 
    d_losses = []
    g_losses = []
    feat_losses = []
    vqvae_losses = []
    g_codes_losses = []
 
    index=0
    for data in self.dataloader:
      index+=1
      
      #Treino com batch real
      real = data['inputs'].to(self.device)
      conditions = data['conditions']
      if conditions is not None:
        conditions = conditions.to(self.device)
          
      ##########      
      # D update
      ##########  
      # Allow D to be updated
          
      for p in self.discriminator.parameters():
        p.requires_grad = True
      
      for _ in range(self.disc_steps):
 
        self.d_optimizer.zero_grad()
        #Forward of the real batch through D
        if self.transforms is not None:
          d_real = self.discriminator(self.transforms(real), conditions)
        else:
          d_real = self.discriminator(real, conditions)
 
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 0.5)
 
        # VQ_VAE reconstructions
        fake, codes = self.vqvae(real)
        # Fake batch goes through d
        if self.transforms is not None:
          d_fake = self.discriminator(self.transforms(fake).detach(), conditions)
        else:
          d_fake = self.discriminator(fake.detach(), conditions)
        
        d_loss = 0
        D_x = 0
        D_G_z1 = 0
        for (_, score_real), (_, score_fake) in zip(d_real, d_fake):
          D_x += score_real.mean().item()
          D_G_z1 += score_fake.mean().item()
          #Cálculo do erro no batch de amostras reais
          d_loss += self.gan_loss(score_real, score_fake, mode='d')
 
        #Calcula os gradientes para o batch
        d_loss.backward()
        #Atualza D
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)     
        self.d_optimizer.step()
     
      #########
      # G update
      #########
      # Stops D from being updated
      for p in self.discriminator.parameters():
        p.requires_grad = False      
      
      self.v_optimizer.zero_grad()
     
      #fake, codes = self.vqvae(real)
      if self.transforms is not None:
        d_fake = self.discriminator(self.transforms(fake), conditions)
      else:
        d_fake = self.discriminator(fake, conditions)
       
      l2_loss, lat_loss, spec_loss = self.vqvae_loss(real, fake, codes, spec_hp=self.spec_hp, mel=self.mel)
      rec_loss = l2_loss + spec_loss
 
      if torch.any(torch.isnan(l2_loss)):# or torch.isinf(tensor):
        print('invalid input detected at iteration ', data['ids'])

      g_loss = 0
      feat_loss = 0
      D_G_z2 = 0
      for (feats_real, _), (feats_fake, score_fake) in zip(d_real, d_fake):
        D_G_z2 += score_fake.mean().item()
        if self.feat_hp != 0.0:
          for feat_real, feat_fake in zip(feats_real, feats_fake):
            feat_loss += F.l1_loss(feat_fake, feat_real.detach())
        # Calculate G loss
        g_loss += self.gan_loss(score_fake=score_fake, mode='g')
      
      g_feat_loss = g_loss + self.feat_hp*feat_loss
      loss_hp = self.get_loss_hp(rec_loss, g_feat_loss, self.vqvae.get_last_layer())
      #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA  
      total_loss = rec_loss + lat_loss + loss_hp*g_feat_loss    
      total_loss.backward()
      # Update G
      torch.nn.utils.clip_grad_norm_(self.vqvae.parameters(), 1.0)     
      self.v_optimizer.step()
           
      # Training statistics    
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        print('{:3d} batches | time: {:5.2f}s | Loss_D: {:5.4f} | Loss_G: {} | VQ_VAE_loss: {} | '
              ' feat_loss: {} | D(x): {:5.4f} | D(G(z)): {:5.4f} / {:5.4f} | loss_hp: {}' 
              .format(index, elapsed, np.mean(d_losses, 0), np.mean(g_losses, 0),
                      np.mean(vqvae_losses, 0), np.mean(feat_losses, 0), D_x, D_G_z1, D_G_z2, loss_hp.item()))              
        start_time = time.time()
 
      # Save losses to plot later
      d_losses.append(d_loss.item())
      g_losses.append(g_loss.item())
      feat_losses.append(feat_loss.item() if type(feat_loss) == torch.Tensor else feat_loss)
      vqvae_losses.append([l2_loss.item(), lat_loss.item(), spec_loss.item() if type(spec_loss) == torch.Tensor else spec_loss])
      
    return np.mean(d_losses, 0), np.mean(g_losses, 0), np.mean(feat_losses, 0), np.mean(vqvae_losses, 0), vqvae_losses
 
  def train(self, epochs, checkpoint_dir, train_gan=False, load=False, log_interval=20, save=True):
    if load:
      self.vqvae.to('cpu')
      self.discriminator.to('cpu')

      checkpoint = torch.load(checkpoint_dir + 'checkpoint.pth',  map_location='cpu')
      # v_state_dict = torch.load(checkpoint['vqvae'], map_location='cpu')
      # d_state_dict = torch.load(checkpoint['discriminator'], map_location='cpu')

      self.vqvae.load_state_dict(checkpoint['vqvae'])
      self.discriminator.load_state_dict(checkpoint['discriminator'])      
 
      self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
      self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    
    self.optimizer_to(self.v_optimizer, self.device)
    self.optimizer_to(self.d_optimizer, self.device)

    self.vqvae.to(self.device)
    self.discriminator.to(self.device)
    
    history = {'train_loss': [], 'train_loss_list': [], 'val_loss': [], 'val_loss_list': []}
    
    best_d_loss = 0.
    best_g_loss = 0.
 
    # list to store generator outputs
    samples=[]
 

    for epoch in range(epochs):
      epoch_start_time = time.time()
      print(f'Epoch {epoch + 1}/{epochs}')
 
      print('-' * 10)
      if train_gan:
        d_loss, g_loss, feat_loss, vqvae_loss, vqvae_loss_list = self.train_epoch_gan(log_interval=log_interval)
        
        history['train_loss'].append(vqvae_loss)
        history['train_loss_list'].append(vqvae_loss_list)
        
        val_vqvae_loss = 0.
        if self.val_dataloader is not None:
          val_vqvae_loss, val_vqvae_loss_list = self.validate(self.val_dataloader)
          history['val_loss'].append(val_vqvae_loss)
          history['val_loss_list'].append(val_vqvae_loss_list)

        print('| End of epoch {:3d}  | time: {:5.2f}s | loss_D: {:5.2f} |'
              'loss_G: {} | feat_loss: {} | loss_VQVAE: {} | val_loss_VQVAE: {}'.format(
              epoch+1, (time.time()-epoch_start_time), d_loss, g_loss,
              feat_loss, vqvae_loss, val_vqvae_loss))
      else:
        vqvae_loss, vqvae_loss_list = self.train_epoch()
        history['train_loss'].append(vqvae_loss)
        history['train_loss_list'].append(vqvae_loss_list)
        
        val_vqvae_loss = 0.
        if self.val_dataloader is not None:
          val_vqvae_loss, val_vqvae_loss_list = self.validate(self.val_dataloader)
          history['val_loss'].append(val_vqvae_loss)
          history['val_loss_list'].append(val_vqvae_loss_list)
 
        print('| End of epoch {:3d}  | time: {:5.2f}s | loss_VQVAE {} |'
              ' val_loss_VQVAE {}  '.format(epoch+1, (time.time()-epoch_start_time),
                                            vqvae_loss, val_vqvae_loss))
      if save:
        checkpoint = { 
              'vqvae': self.vqvae.state_dict(),
              'discriminator': self.discriminator.state_dict(),
              'v_optimizer': self.v_optimizer.state_dict(),
              'd_optimizer': self.d_optimizer.state_dict()}
          
        torch.save(checkpoint, checkpoint_dir + 'checkpoint.pth')
 
      fake = self.evaluate(self.fixed_real, self.fixed_conditions)
      samples.append(fake)
                   
    return samples, history

  def optimizer_to(self, optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
 
  def evaluate(self, real, conditions):
    self.vqvae.eval()
    if conditions is not None:
      conditions = conditions.to(self.device)
    with torch.no_grad():
      fake= self.vqvae(real)[0].detach().cpu()
      return fake
  
  def validate(self, val_dataloader):
    vqvae_losses = []
    self.vqvae.eval()
    with torch.no_grad():
      for data in val_dataloader:
        real = data['inputs'].to(self.device)
        conditions = data['conditions']
        if conditions is not None:
          conditions = conditions.to(self.device)
        
        fake, codes = self.vqvae(real)
        l2_loss, lat_loss, spec_loss = self.vqvae_loss(real, fake, codes, spec_hp=self.spec_hp, mel=self.mel)
        vqvae_losses.append([l2_loss.item(), lat_loss.item(), spec_loss.item() if type(spec_loss) == torch.Tensor else spec_loss])

    return np.mean(vqvae_losses, 0), vqvae_losses
    
  def save_model(self, checkpoint_dir):
      torch.save(self.vqvae.state_dict(), checkpoint_dir + 'vqvae_state.bin')    
      torch.save(self.discriminator.state_dict(), checkpoint_dir + 'discriminator_state.bin')
