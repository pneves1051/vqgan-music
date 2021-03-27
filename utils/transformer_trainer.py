import torch

class TransformerTrainer():
  def __init__(self, model, dataloader, valid_dataloader, loss_fn, device, output_length, lr):
    self.model = model
    self.dataloader = dataloader
    self.valid_dataloader = valid_dataloader
    self.loss_fn = loss_fn
    self.device=device
    self.output_length = output_length
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
    
  def train_epoch(self):
    self.model = self.model.train()
       
    losses = []
    correct_predictions = 0
    start_time = time.time()
  
    for index, data in enumerate(self.dataloader):
      inputs = data['inputs'].to(device)
      targets = data['targets'].to(device)
      self.optimizer.zero_grad()

      mask = self.model.generate_square_subsequent_mask(inputs.size(1)).to(device)
      outputs = self.model(inputs, mask)
      
      loss = self.loss_fn(outputs, targets)
      loss.backward()
      #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
      self.optimizer.step()

      preds = torch.argmax(F.softmax(outputs, dim=1), dim = 1)
      correct_predictions += torch.sum(preds == targets)
      #print(set(preds.reshape(-1).tolist()))
      losses.append(loss.item())
                  
      log_interval = 20
      if index % log_interval == 0 and index > 0:
        elapsed = time.time() - start_time
        current_loss = np.mean(losses)
        print('| {:5d} of {:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
              'loss {:5.2f} | acc {:8.4f}'.format(
              index, len(self.dataloader), self.scheduler.get_last_lr()[0], elapsed*1000/log_interval,
              current_loss,  correct_predictions.double() /(index*self.dataloader.batch_size*targets.shape[1])))
        start_time = time.time()


    train_acc =  correct_predictions.double() /(len(self.dataloader)*self.dataloader.batch_size*self.output_length)
    train_loss = np.mean(losses)
    return train_acc, train_loss

  def train(self, EPOCHS, checkpoint_dir, validate = False):
    history = defaultdict(list)
    best_accuracy = 0

    valid_acc =0
    valid_loss = 10
    for epoch in range(EPOCHS):
      epoch_start_time = time.time()
      print(f'Epoch {epoch + 1}/{EPOCHS}')

      print('-' * 10)

      train_acc, train_loss = self.train_epoch()
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      if validate:
        valid_acc, valid_loss = self.evaluate(self.valid_dataloader)

      print('| End of epoch {:3d}  | time: {:5.2f}s | train loss {:5.2f} | '
            'train ppl {:8.2f} | \n train accuracy {:5.2f} | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid accuracy {:5.2f} |'.format(
            epoch+1, (time.time()-epoch_start_time), train_loss, math.exp(train_loss), train_acc,
            valid_loss, math.exp(valid_loss), valid_acc))


      if validate and valid_acc > best_accuracy :
        torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
        best_accuracy = valid_acc
      elif train_acc > best_accuracy:
        torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
        best_accuracy = train_acc

      self.scheduler.step()
    
  def evaluate(self, eval_dataloader):
    self.model.eval()
    eval_losses = []
    eval_correct_predictions = 0
    with torch.no_grad():
      for index, data in enumerate(eval_dataloader):
        inputs = data['inputs']
        targets = data['targets']

        mask = self.model.generate_square_subsequent_mask(inputs.size(1)).to(device)
        outputs = self.model(inputs, mask)

        eval_loss = self.loss_fn(outputs, targets)
        
        preds = torch.argmax(F.softmax(outputs, dim=1), dim = 1)
        eval_correct_predictions += torch.sum(preds == targets)
        eval_losses.append(eval_loss.item())
                
        eval_acc = eval_correct_predictions.double() /(len(eval_dataloader)*eval_dataloader.batch_size*targets.shape[1])
        eval_loss = np.mean(eval_losses)
        
    return eval_acc, eval_loss

  def save_model(self, checkpoint_dir):
    torch.save(self.model.state_dict(), checkpoint_dir + 'best_transformer_state.bin')
