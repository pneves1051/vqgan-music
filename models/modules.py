

class ResLayer(nn.Module):
  def __init__(self, filters, dilation, leaky=False):
    super(ResLayer, self).__init__()
    padding = dilation
    self.conv = nn.Sequential(
                  nn.Conv1d(filters, filters, 3, dilation=dilation, padding=padding),
                  nn.BatchNorm1d(filters),
                  nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                  nn.Conv1d(filters, filters, 3, padding=1),
                  nn.BatchNorm1d(filters))
    self.relu =  nn.LeakyReLU(0.2) if leaky else nn.ReLU()
       
  def forward(self, inputs):
    res_x = self.conv(inputs)
    res_x += inputs
    res_x = self.relu(res_x)
    return res_x

class ResBlock(nn.Module):
  def __init__(self, filters, dilations, depth, leaky=False):
    super(ResBlock, self).__init__()
    self.res_block = nn.Sequential(*[ResLayer(filters, dilations[i], leaky) for i in range(depth)])

  def forward(self, inputs):
    output = self.res_block(inputs)
    return output
  
  class VQVAEEncoder(nn.Module):
  def __init__(self, first_filter, last_filter, num_filters, depth, leaky=False):
    super(VQVAEEncoder,self).__init__()
    dilations = [3**i for i in range(depth)]

    # possível camada para condicionamento
    # self.cond = nn.AdaptiveMaxpool()

    kernel_size = 3
    stride=4
    padding = (kernel_size-stride)//2

    self.causal_conv = nn.Conv1d(first_filter, num_filters[0], 3, padding=1)

    res_list = []
    for i in range(len(num_filters)):
      res_list.extend([ResBlock(num_filters[i], dilations, depth, leaky),
                      nn.Conv1d(num_filters[i], num_filters[i], kernel_size, stride=stride, padding=padding),
                      nn.BatchNorm1d(num_filters[i]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU()
                      ])
    self.res_layers =  nn.Sequential(*res_list)

    # camadas de pós processamento
    self.post = nn.Sequential(nn.Conv1d(num_filters[-1], num_filters[-1], 3, padding=1),
                              nn.BatchNorm1d(num_filters[-1]),
                              nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                              nn.Conv1d(num_filters[-1], last_filter, 3, padding=1))
    
  def forward(self, inputs):
    x = self.causal_conv(inputs)
    x = self.res_layers(x)
    output = self.post(x)
    return output

class VQVAEDecoder(nn.Module):
  def __init__(self, first_filter, last_filter, num_filters, depth, leaky=False):
    super(VQVAEDecoder,self).__init__()
    dilations = [3**i for i in range(depth)]
    dilations = dilations[::-1]
    self.causal_conv = nn.Conv1d(first_filter, num_filters[0], 3, padding=1)

    kernel_size =3
    stride = 4
    output_padding = 0
    padding = (kernel_size + output_padding-stride)//2

    res_list = []
    for i in range(len(num_filters)):
      res_list.extend([ResBlock(num_filters[i], dilations, depth, leaky),
                      nn.ConvTranspose1d(num_filters[i], num_filters[i], kernel_size, stride=stride, padding=padding),
                      nn.BatchNorm1d(num_filters[i]),
                      nn.LeakyReLU(0.2) if leaky else nn.ReLU()])
    self.res_layers =  nn.Sequential(*res_list)

    # camadas de pós processamento
    self.post = nn.Sequential(
                          nn.Conv1d(num_filters[-1], num_filters[-1], 3, padding=1),
                          nn.BatchNorm1d(num_filters[-1]),
                          nn.LeakyReLU(0.2) if leaky else nn.ReLU(),
                          nn.Conv1d(num_filters[-1], last_filter, 3, padding=1))
    
  def forward(self, inputs):
    x = self.causal_conv(inputs)
    x = self.res_layers(x)
    output = self.post(x)
    return output
  
  
