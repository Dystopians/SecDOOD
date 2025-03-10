import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class HyperNetwork(nn.Module):

    def __init__(self, batchsize = 16, f_size = 3, z_dim = 64, out_size=16, in_size=16, hidden = 32):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.hidden = hidden
        self.batchsize = batchsize
        self.bn = nn.BatchNorm1d(self.in_size)

        self.w1 = Parameter(torch.fmod(torch.randn((self.hidden, self.out_size*self.f_size*self.f_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.in_size, self.out_size * self.f_size*self.f_size*self.f_size)).cuda(),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.hidden)).cuda(),2))
        # 2048 * 2048 * 32
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.hidden)).cuda(),2))

    def forward(self, z):
        # z = batch*512 / w2 = 512 * in_size/ batch * in_size -> h_in
        # w1 = in_size * self.out_size * self.in_size * self.f_size^2
        #
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # self.batchsize = z.shape[0]
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        h_in = torch.matmul(z, self.w2) + self.b2

        # h_in z_dim *
        h_in = h_in.view(-1, self.in_size, self.hidden)
        # in_size * hidden
        h_final = torch.matmul(h_in, self.w1) + self.b1
        # batch normalization
        h_final = self.bn(h_final)
        kernel = h_final.view(-1, self.out_size, self.in_size, self.f_size, self.f_size, self.f_size)
        '''
        h_linear_w = torch.matmul(h_in, self.w3) + self.b3
        h_linear_b = torch.matmul(h_in,self.w4)+self.b4

        linear_w = h_linear_w.view(self.in_size, self.numclasses)
        '''

        return kernel#, (linear_w,h_linear_b)

class HyperNetwork_Linear(nn.Module):

    def __init__(self, batchsize=16, hidden=32, x_fast_slow=768, numclasses=25):
        super(HyperNetwork_Linear, self).__init__()
        self.z_dim = x_fast_slow
        self.in_size = x_fast_slow
        self.hidden = hidden
        self.numclasses = numclasses
        self.batchsize = batchsize
        self.bn = nn.BatchNorm1d(self.in_size)
        self.bnb = nn.BatchNorm1d(self.numclasses)
        # h_in = self.bn(torch.matmul(z, self.w2) + self.b2)
        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.hidden)).cuda(), 2))
        # 2048 * 2048 * 32
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.hidden)).cuda(), 2))

        self.w3 = Parameter(torch.fmod(torch.randn((self.hidden, self.numclasses)).cuda(), 2))
        self.b3 = Parameter(torch.fmod(torch.randn((self.in_size, self.numclasses)).cuda(), 2))

        self.w4 = Parameter(torch.fmod(torch.randn((self.in_size, self.numclasses)).cuda(), 2))
        self.b4 = Parameter(torch.fmod(torch.randn((self.numclasses)).cuda(), 2))

        self.w5 = Parameter(torch.fmod(torch.randn((self.z_dim, self.z_dim * self.numclasses)).cuda(), 2))
        self.b5 = Parameter(torch.fmod(torch.randn((self.z_dim * self.numclasses)).cuda(), 2))


    def forward(self, z):
        # z = batch*512 / w2 = 512 * in_size/ batch * in_size -> h_in
        # w1 = in_size * self.out_size * self.in_size * self.f_size^2
        #
        # import pdb;pdb.set_trace()
        # batch normalization
        # z = self.bn(z)
        h_in = (torch.matmul(z, self.w5) + self.b5)
        h_in = h_in.view(-1,self.in_size, self.numclasses)
        # ReLU
        # h_in = F.relu(h_in)
        h_in = self.bn(h_in)
        h_linear_w = h_in
        # h_linear_w = F.relu(h_linear_w)
        # h_linear_w = self.bn(h_linear_w)
        h_linear_b = torch.matmul(z, self.w4) + self.b4
        # h_linear_b = self.bnb(h_linear_b)
        # import pdb;pdb.set_trace()
        linear_w = h_linear_w.view(-1, self.in_size, self.numclasses)
        # h_linear_b = F.relu(h_linear_b)
        h_linear_b = self.bnb(h_linear_b)
        # import pdb;pdb.set_trace()
        return linear_w, h_linear_b

class HyperNetwork_Li(nn.Module):

    def __init__(self, hidden=32, z_dim=2816, label_size=25, batch_size=16, zoom_dim=2816, net_type="twoM+bias"):
        super(HyperNetwork_Li, self).__init__()
        self.z_dim = z_dim
        self.hidden = hidden
        self.label_size = label_size
        self.batch_size = batch_size
        self.zoom_dim = zoom_dim
        self.net_type = net_type

        if self.net_type == 'twoM+bias':
            self.w1 = Parameter(torch.randn((self.hidden, self.zoom_dim * self.label_size)).cuda())  # 128, 96*1001
            self.b1 = Parameter(torch.randn((self.label_size * self.zoom_dim)).cuda())

            self.w2 = Parameter(torch.randn((self.z_dim, self.hidden)).cuda())  # 192,128
            self.b2 = Parameter(torch.randn((self.hidden)).cuda())

            self.b_hyper1 = Parameter(torch.randn(self.hidden, self.label_size)).cuda()

        if self.net_type == 'oneM+bias':
            self.w3 = Parameter(
                torch.randn((self.z_dim, self.zoom_dim * self.label_size)).cuda())  # 192*96*label_size
            self.b3 = Parameter(torch.randn((self.zoom_dim * self.label_size)).cuda())

            self.b_hyper2 = Parameter(torch.randn(self.z_dim, self.label_size)).cuda()

        if self.net_type == 'onebias':
            self.b_hyper3 = Parameter(torch.randn(self.z_dim, self.label_size)).cuda()

        if self.net_type == 'twobias':
            self.w4 = Parameter(torch.randn((self.z_dim, self.hidden)).cuda())  # 192*96*label_size
            self.b4 = Parameter(torch.randn((self.hidden)).cuda())

            self.b_hyper4 = Parameter(torch.randn(self.hidden, self.label_size)).cuda()

    def forward(self, z):

        # print("Z size",z.shape)
        if self.net_type == 'twoM+bias':
            h_in = torch.matmul(z, self.w2) + self.b2
            h_in = h_in.view(-1, self.hidden)

            h_final = torch.matmul(h_in, self.w1) + self.b1
            kernel = h_final.view(-1, self.zoom_dim, self.label_size)

            bias = torch.matmul(h_in, self.b_hyper1)

        if self.net_type == 'oneM+bias':
            h_final = torch.matmul(z, self.w3) + self.b3
            kernel = h_final.view(-1, self.zoom_dim, self.label_size)

            bias = torch.matmul(z, self.b_hyper2)

        if self.net_type == 'onebias':
            bias = torch.matmul(z, self.b_hyper3)
            return None, bias

        if self.net_type == 'twobias':
            h_in = torch.matmul(z, self.w4) + self.b4

            bias = torch.matmul(h_in, self.b_hyper4)
            return None, bias

        # print("bias",bias.shape)

        return kernel, bias

class Embedding(nn.Module):

    def __init__(self, input_size=192, output_size=1501, batch_size=16, net_type='twoM+bias'):
        super(Embedding, self).__init__()
        # self.hyper_linear=nn.Linear(input_size, output_size, bias=False)
        self.batch_size = batch_size
        self.net_type = net_type

    def forward(self, x, para, bias):
        if self.net_type == 'twoM+bias' or self.net_type == 'oneM+bias':
            outputs = (torch.bmm(x.unsqueeze(1),
                                 para).squeeze() + bias)  # 32*1*96;32*96*label_size  + 32*label_size------> 32*1*label_size

        if self.net_type == 'onebias' or self.net_type == 'twobias':
            outputs = x + bias
        # print("outputs",outputs.shape)
        return outputs

    '''
    
class HyperNetwork_Linear(nn.Module):

    def __init__(self, batchsize=16, hidden=32, x_fast_slow=768, numclasses=25):
        super(HyperNetwork_Linear, self).__init__()
        self.z_dim = x_fast_slow
        self.in_size = x_fast_slow
        self.hidden = hidden
        self.numclasses = numclasses
        self.batchsize = batchsize
        self.bn = nn.BatchNorm1d(self.in_size)
        self.bnb = nn.BatchNorm1d(self.numclasses)
        # h_in = self.bn(torch.matmul(z, self.w2) + self.b2)
        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.hidden)).cuda(), 2))
        # 2048 * 2048 * 32
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.hidden)).cuda(), 2))

        self.w3 = Parameter(torch.fmod(torch.randn((self.hidden, self.numclasses)).cuda(), 2))
        self.b3 = Parameter(torch.fmod(torch.randn((self.in_size, self.numclasses)).cuda(), 2))

        self.w4 = Parameter(torch.fmod(torch.randn((self.in_size, self.numclasses)).cuda(), 2))
        self.b4 = Parameter(torch.fmod(torch.randn((self.numclasses)).cuda(), 2))


    def forward(self, z):
        # z = batch*512 / w2 = 512 * in_size/ batch * in_size -> h_in
        # w1 = in_size * self.out_size * self.in_size * self.f_size^2
        #
        # import pdb;pdb.set_trace()
        # batch normalization
        z = self.bn(z)
        h_in = (torch.matmul(z, self.w2) + self.b2)
        h_in = h_in.view(-1,self.in_size, self.hidden)
        # ReLU
        # h_in = F.relu(h_in)
        h_in = self.bn(h_in)
        h_linear_w = torch.matmul(h_in, self.w3) + self.b3
        # h_linear_w = F.relu(h_linear_w)
        h_linear_w = self.bn(h_linear_w)
        h_linear_b = torch.matmul(z, self.w4) + self.b4
        # h_linear_b = self.bnb(h_linear_b)
        # import pdb;pdb.set_trace()
        linear_w = h_linear_w.view(-1, self.in_size, self.numclasses)
        # h_linear_b = F.relu(h_linear_b)
        h_linear_b = self.bnb(h_linear_b)
        # import pdb;pdb.set_trace()
        return linear_w, h_linear_b
    
    '''


