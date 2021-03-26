class Transformer(nn.Module):
  def __init__(self, emb_dim, num_embeddings, nhead, nlayers, codebook, dropout = 0.1):
    super(Transformer, self).__init__()
    self.model_type = 'Transformer'
    self.num_embeddings = num_embeddings
    self.emb_dim = emb_dim
    
    self.src_mask = None
    self.pos_encoder = PositionalEncoding(emb_dim, dropout)
    self.embedding = nn.Embedding(num_embeddings, emb_dim)
    #self.embedding.load_state_dict({'weight': codebook})
    #self.embedding.requires_grad=False

    encoder_layer = nn.TransformerEncoderLayer(emb_dim, nhead, dropout=dropout)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
    self.decoder = nn.Linear(emb_dim, num_embeddings)
    
    #self.upsampler1 = nn.Linear(num_embeddings, emb_dim)
    #self.upsampler2 = nn.Upsample(scale_factor = int(bottom_len/top_len))
    #self.downsampler = nn.Linear(emb_dim+emb_dim, emb_dim)

    self.init_weights()
    
  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

  def init_weights(self):
    initrange =0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-initrange, initrange)
    
  def forward(self, input, mask):
    
    src = self.embedding(input)

    src = src.permute(1,0,2)*math.sqrt(self.emb_dim)
    src = self.pos_encoder(src)
        
    src = self.transformer_encoder(src, mask)
    output = self.decoder(src)

    #top_cond = self.upsampler1(top_output.permute(1, 0, 2))
    #top_cond = self.upsampler2(top_cond.permute(0, 2, 1)).permute(2, 0, 1)

    return output.permute(1, 2, 0)
