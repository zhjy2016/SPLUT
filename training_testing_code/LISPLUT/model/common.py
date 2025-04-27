
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import numpy as np


class _baseq(torch.autograd.Function):
    @staticmethod

    def forward(ctx, x, steps):
        y_step_ind=torch.floor(x / steps)
        y = y_step_ind * steps
        return y
        # return y_step_ind

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None




class BASEQ(nn.Module):
    def __init__(self, lvls,activation_range):
        super(BASEQ, self).__init__()
        self.lvls = lvls
        self.activation_range = activation_range
        self.steps = 2 * activation_range / self.lvls

    def forward(self, x):
        x=(((-x - self.activation_range).abs() - (x - self.activation_range).abs()))/2.0
        x[x > self.activation_range-0.1*self.steps] =self.activation_range-0.1*self.steps
        return _baseq.apply(x, self.steps)




class LTQ_R(nn.Module):
    def __init__(self, n_val, max_range):
        super(LTQ_R, self).__init__()
        #带数据反传
        # init_range = 2.0
        # self.n_val = 2 ** num_bits - 1
        init_range = 2.0 * max_range
        self.n_val = n_val
        self.interval = init_range / self.n_val #平均宽度
        # self.start = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.start = nn.Parameter(torch.Tensor([-1.0* max_range]), requires_grad=True)
        self.input_interval = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)

    def forward(self, x):
        x = x * self.scale1

        self.x_quant = x
        x_forward = x
        x_backward = x
        step_right = self.minusone + 0.0

        a_pos = torch.where(self.input_interval > self.eps, self.input_interval, self.eps) #获取不小于0的interval
        # print('self.scale1',self.scale1.data,'self.scale2',self.scale2.data)
        # print('a_pos',a_pos.data)
        # model_G_A.inltq1

        for i in range(self.n_val):
            step_right += self.interval #量化输出 区间右侧
            
            # print(self.tab_value)

            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                # pdb.set_trace()
                self.x_quant = torch.where(x > thre_forward, thre_forward, self.minusone)
                x_forward = torch.where(x > thre_forward, step_right, self.minusone)
                # print(x,thre_forward,self.x_quant,x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, self.minusone)
            else:
                thre_forward += a_pos[i-1] / 2 +  a_pos[i] / 2
                thre_backward += a_pos[i-1] # 量化输入 区间左侧
                self.x_quant = torch.where(x > thre_forward, thre_forward, self.x_quant)
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                # print(x,thre_forward,self.x_quant,x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)
            # self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval 
            ## self.interval/a_pos [i] 从输入的区间映射到输出区间
            ## (x - thre_backward) 减去输入区间左侧
            ## step_right - self.interval 加上输出区间左侧

            # print('==================i',i)
            # print('step_right',step_right.data)
            # print('thre_forward',thre_forward.data,'thre_backward',thre_backward.data)
            # print('x',x[48,0,20,20].data,'x_forward',x_forward[48,0,20,20].data,'x_backward',x_backward[48,0,20,20].data)
            # print('*i',i,'*step_right',step_right.data,'*thre_forward',thre_forward.data,'*thre_backward',thre_backward.data,'*x',x[48,0,20,20].data,'*x_forward',x_forward[48,0,20,20].data,'*x_backward',x_backward[48,0,20,20].data)
        # pdb.set_trace()

        thre_backward += a_pos[i] 
        x_backward = torch.where(x > thre_backward, self.one, x_backward) #值，hard

        out = x_forward.detach() + x_backward - x_backward.detach()
        # pdb.set_trace()

        self.x_quant = self.x_quant / self.scale1
        self.feat_value = x_forward.detach()
        out = out * self.scale2
        # pdb.set_trace()
        wgt = torch.histc(x_forward.data, self.n_val).float().view(1, 1, -1, 1).sqrt() + 1e-5
        # pdb.set_trace()
        a_pos = a_pos.data + (a_pos - a_pos.data) / wgt * x_forward.numel() / self.n_val

        return out
