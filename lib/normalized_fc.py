import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module

class CosineLinear(Module):
    def __init__(self, in_features, out_features, device, bias=True, eta=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if eta:
            self.eta = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('eta', None)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_features,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        ######################Initial#########################
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        ######################################################        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
        if self.eta is not None:
            self.eta.data.fill_(1) #for initializaiton of eta

    def forward(self, input):       
        if self.bias is not None:
            input = torch.cat((input, (torch.ones(len(input),1)).to(self.device)), dim=1)
            concat_weight = torch.cat((self.weight, self.bias), dim=1)
            out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(concat_weight, p=2, dim=1))
            
        else:
            out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.eta is not None:
            out = self.eta * out
        return out
