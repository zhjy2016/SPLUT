# SP-LUT-M-denseconv-outquant256-noskip


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob
from model.common import LTQ_R
from quant_ops import quant_lookup
import pdb

from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter

EXP_NAME = "base-M-org-re42-palq44-DataRescale-v3"

UPSCALE = 4     # upscaling factor

NB_BATCH = 32        # mini-batch
CROP_SIZE = 48       # input LR training patch size

# START_ITER = -1      # Set 0 for from scratch, else will load saved params and trains further
# PATH_LOAD = '/home/zjy/codes/SPLUT/training_testing_code/checkpoint/befor522/base-M-org_0.0001_4_0.001'
START_ITER = 1      # Set 0 for from scratch, else will load saved params and trains further
PATH_LOAD = '/home/zjy/codes/SPLUT/training_testing_code/checkpoint/base-M-org-re42-palq44-DataRescale_0.0001_4_0.001'
NB_ITER = 5000000    # Total number of training iterations

I_DISPLAY = 1000     # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

TRAIN_DIR = '/home/zjy/datas/DIV2K/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = '/home/zjy/datas/benchmark/Set5/'      # Validation images

LR_G = 1e-4         # Learning rate for the generator

QQ=2**4
OUT_NUM=4
TAU=0.001
VERSION = "{}_{}_{}_{}".format(EXP_NAME,LR_G,OUT_NUM,TAU)


### Tensorboard for monitoring ###
writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))

class _baseq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, steps):
        y_step_ind=torch.floor(x / steps)
        y = y_step_ind * steps
        return y

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


############### Basic Convolutional Layers ###############
class Conv(nn.Module):
    """ 2D convolution w/ MSRA init. """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class DenseConv(nn.Module):
    """ Dense connected Conv. with activation. """

    def __init__(self, in_nf, nf=64):
        super(DenseConv, self).__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv(in_nf, nf, 1)

    def forward(self, x):
        feat = self.act(self.conv1(x))
        out = torch.cat([x, feat], dim=1)
        return out

############### MuLUT Blocks ###############
class LUTBlock(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """
    def __init__(self, mode, nf, upscale=1, dense=True):
        super(LUTBlock, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale
        

        if mode == '1x2x2':
            self.conv1 = Conv(1, nf, (2, 2))
        elif mode == '2x1x2':
            self.conv1 = Conv(2, nf, (1, 2))
        elif mode == '2x2x1':
            self.conv1 = Conv(2, nf, (2, 1))
        elif mode == '1x1x1':
            self.conv1 = Conv(1, nf, 1)
        elif mode == '2x1x1':
            self.conv1 = Conv(2, nf, 1)
        elif mode == '4x1x1':
            self.conv1 = Conv(4, nf, 1)
        elif mode == '8x1x1':
            self.conv1 = Conv(8, nf, 1)
        else:
            raise AttributeError


        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            if self.upscale > 1:
                self.pixel_shuffle = nn.PixelShuffle(upscale)
                self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
            else:
                self.conv6 = Conv(nf * 5, 8, 1)
        else:
            raise AttributeError
        
        

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # pdb.set_trace()
        x = torch.tanh(self.conv6(x))
        # if self.upscale > 1:
            # x = self.pixel_shuffle(x)
        return x


### A lightweight deep network ###
class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale
        self.out_channal= OUT_NUM

        self.lvls = 16
        # self.inquant= BASEQ(self.lvls ,1.0)
        self.inLTQ_R11 = LTQ_R(self.lvls ,1.0)
        self.inLTQ_R12 = LTQ_R(self.lvls ,1.0)
        self.inLTQ_R21 = LTQ_R(self.lvls ,1.0)
        self.inLTQ_R22 = LTQ_R(self.lvls ,1.0)

        nf=64

        

        #cwh
        self.lut1st= LUTBlock('1x2x2', nf, upscale=1, dense=True)

        self.lut1st_1x1= LUTBlock('1x1x1', nf, upscale=1, dense=True)

        self.lut2nd_221= LUTBlock('2x2x1', nf, upscale=1, dense=True)

        self.lut2nd_212= LUTBlock('2x1x2', nf, upscale=1, dense=True)

        self.lut2nd_1x1= LUTBlock('1x1x1', nf, upscale=1, dense=True)

        self.lut3rd_221= LUTBlock('2x2x1', nf, upscale=self.upscale, dense=True)

        self.lut3rd_212= LUTBlock('2x1x2', nf, upscale=self.upscale, dense=True)

        self.lut3rd_1x1= LUTBlock('1x1x1', nf, upscale=self.upscale, dense=True)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def FirstLayer(self, x_in):
        x = self.lut1st(x_in)
        return x

    def SecondLayer(self, x):
        x2nd_221_in = self.inLTQ_R11((F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect'))/2.0)
        x2nd_221 = self.lut2nd_221(x2nd_221_in)
        x2nd_212_in = self.inLTQ_R12((F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect'))/2.0)
        x2nd_212 = (self.lut2nd_212(x2nd_212_in))
        # print(x2nd_221_in.max(),x2nd_221_in.min())
        x=(x2nd_221+x2nd_212)/2.0
        return x

    def LastLayer(self, x):
        x3rd_221_in=self.inLTQ_R21((F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect'))/2.0)
        x3rd_221 = self.lut3rd_221(x3rd_221_in)
        x3rd_212_in=self.inLTQ_R22((F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect'))/2.0)
        x3rd_212 = self.lut3rd_212(x3rd_212_in)
        # print(x3rd_221_in.max(),x3rd_221_in.min())
        # pdb.set_trace()
        x=(x3rd_221+x3rd_212)/2.0
        return x

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)

        x1 = self.FirstLayer(x_in)+x_in[:,:,:H-1,:W-1]
        x2 = self.SecondLayer(x1)+x1
        x = self.LastLayer(x2)+x_in[:,:,:H-1,:W-1]

        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))
        return x


class SPNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SPNet, self).__init__()

        self.upscale=upscale
        self.inquant_1x1= BASEQ(256 ,1.0)
        
        self.branchm=SRNet(upscale=upscale)
        self.branchl=SRNet(upscale=upscale)
        
    def SpaceInference(self, model,batch):
        batch_S1 = model(F.pad(batch, (0,1,0,1), mode='reflect'))
        batch_S = torch.clamp(batch_S1,-1,1)
        return batch_S
    
    def FloatToData(self, batch_L):
        batch_L225 = torch.floor(batch_L*255)
        batch_L_A= batch_L225// QQ
        batch_L_B= batch_L225 % QQ
        batch_L_A=batch_L_A/QQ
        batch_L_B=batch_L_B/QQ
        return batch_L_A,batch_L_B

    def forward_mm(self, x_in):
        batch_L_A , batch_L_B=self.FloatToData(x_in)

        batch_S_A=self.SpaceInference(self.branchm,batch_L_A)
        batch_S_B=self.SpaceInference(self.branchl,batch_L_B)
        batch_S_all = batch_S_A+batch_S_B
        
        return batch_S_all

    def forward(self, x_in):
        x_in = F.pad(x_in, (0,1,0,1), mode='reflect')
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)
        batch_L_m , batch_L_l=self.FloatToData(x_in)
        
        x1_m = self.branchm.FirstLayer(batch_L_m)+batch_L_m[:,:,:H-1,:W-1] + self.branchm.lut1st_1x1(batch_L_l[:,:,:H-1,:W-1])
        x1_l= self.branchl.FirstLayer(batch_L_l)+batch_L_l[:,:,:H-1,:W-1] + self.branchl.lut1st_1x1(batch_L_m[:,:,:H-1,:W-1])

        # x2_m = self.branchm.SecondLayer(x1_m)+x1_m + self.branchm.lut2nd_1x1(self.inquant_1x1(x1_l.mean(dim=1,keepdim=True)))
        # x2_l = self.branchl.SecondLayer(x1_l)+x1_l + self.branchl.lut2nd_1x1(self.inquant_1x1(x1_m.mean(dim=1,keepdim=True)))
        x2_m = self.branchm.SecondLayer(x1_m)+x1_m + self.branchm.lut2nd_1x1(batch_L_l.mean(dim=1,keepdim=True)[:,:,:H-1,:W-1])
        x2_l = self.branchl.SecondLayer(x1_l)+x1_l + self.branchl.lut2nd_1x1(batch_L_m.mean(dim=1,keepdim=True)[:,:,:H-1,:W-1])

        # x_m = self.branchm.LastLayer(x2_m)+batch_L_m[:,:,:H-1,:W-1] + self.branchm.lut3rd_1x1(self.inquant_1x1(x2_l.mean(dim=1,keepdim=True)))
        # x_l = self.branchl.LastLayer(x2_l)+batch_L_l[:,:,:H-1,:W-1] + self.branchl.lut3rd_1x1(self.inquant_1x1(x2_m.mean(dim=1,keepdim=True)))
        x_m = self.branchm.LastLayer(x2_m)+batch_L_m[:,:,:H-1,:W-1] + self.branchm.lut3rd_1x1(batch_L_l.mean(dim=1,keepdim=True)[:,:,:H-1,:W-1])
        x_l = self.branchl.LastLayer(x2_l)+batch_L_l[:,:,:H-1,:W-1] + self.branchl.lut3rd_1x1(batch_L_m.mean(dim=1,keepdim=True)[:,:,:H-1,:W-1])

        # pdb.set_trace()



        x_m = self.branchm.pixel_shuffle(x_m)
        x_l = self.branchl.pixel_shuffle(x_l)


        x_m = torch.clamp(x_m,-1,1)
        x_l = torch.clamp(x_l,-1,1)
        x = x_m + x_l

        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))

        batch_S_all = x
        
        return batch_S_all



if __name__ == '__main__':
    model_G = SPNet(upscale=UPSCALE).cuda()
    ## Optimizers
    params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
    opt_G = optim.Adam(params_G,lr=LR_G)

    ## Load saved params
    if START_ITER > 0:
        lm = torch.load('{}/model_G_All_best.pth'.format(PATH_LOAD))
        model_G.load_state_dict(lm.state_dict(), strict=False)
    if START_ITER < 0:
        lm = torch.load('{}/model_G_A_best.pth'.format(PATH_LOAD))
        model_G.branchm.load_state_dict(lm.state_dict(), strict=False)

        lm = torch.load('{}/model_G_B_best.pth'.format(PATH_LOAD))
        model_G.branchl.load_state_dict(lm.state_dict(), strict=False)
    

    # Training dataset
    Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K( 
                                    datadir = TRAIN_DIR,
                                    crop_size = CROP_SIZE, 
                                    crop_per_image = NB_BATCH//4,
                                    out_batch_size = NB_BATCH,
                                    scale_factor = UPSCALE,
                                    shuffle=True))
    Iter_H.start(max_q_size=16, workers=4)

    ## Prepare directories
    if not isdir('checkpoint'):
        mkdir('checkpoint')
    if not isdir('result'):
        mkdir('result')
    if not isdir('checkpoint/{}'.format(str(VERSION))):
        mkdir('checkpoint/{}'.format(str(VERSION)))
    if not isdir('result/{}'.format(str(VERSION))):
        mkdir('result/{}'.format(str(VERSION)))

    ## Some preparations 
    print('===> Training start')
    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    def SaveCheckpoint(best=False):
        if best:
            torch.save(model_G, 'checkpoint/{}/model_G_All_best.pth'.format(str(VERSION)))
            torch.save(opt_G, 'checkpoint/{}/opt_G_best.pth'.format(str(VERSION)))
        else:
            torch.save(model_G, 'checkpoint/{}/model_G_All_latest.pth'.format(str(VERSION)))
            torch.save(opt_G, 'checkpoint/{}/opt_G_latest.pth'.format(str(VERSION)))
        print("Checkpoint saved")



    ### TRAINING
    psnr_best=0.0
    for i in tqdm(range(START_ITER+1, NB_ITER+1)):

        model_G.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = Iter_H.dequeue()
        
        batch_H = Variable(torch.from_numpy(batch_H)).cuda()      # BxCxHxW, range [0,1]
        batch_L = Variable(torch.from_numpy(batch_L)).cuda()      # BxCxHxW, range [0,1]


        dT += time.time() - st
        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        batch_S_all = model_G(batch_L)
        batch_S_all = torch.clamp(batch_S_all,0,1)

        loss_Pixel = torch.mean( ((batch_S_all - batch_H)**2))
        loss_G = loss_Pixel

        # Update
        loss_G.backward()
        # print(model_G.grad)
        # model_G.branchm.lut1st.conv1.conv.weight.grad
        # (Pdb) model_G.branchm.lut1st.conv1.conv.weight.grad[0,]
        # tensor([[[-7.9189e-09,  3.0511e-07],
        #  [ 4.1915e-08, -9.4419e-08]]], device='cuda:0')
        # pdb.set_trace()
        opt_G.step()
        rT += time.time() - st

        # For monitoring
        accum_samples += NB_BATCH
        l_accum[0] += loss_Pixel.item()

        ## Show information
        if i % I_DISPLAY == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)
            print("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.

        ## Save models
        if i % I_SAVE == 0:
            SaveCheckpoint()

        ## Validation
        if i % I_VALIDATION == 0:
            with torch.no_grad():
                model_G.eval()

                # Test for validation images
                files_gt = glob.glob(VAL_DIR + '/HR/*.png')
                files_gt.sort()
                files_lr = glob.glob(VAL_DIR + '/LR_bicubic/X4/*.png')
                files_lr.sort()

                psnrs = []
                lpips = []

                for ti, fn in enumerate(files_gt):
                    # Load HR image
                    tmp = _load_img_array(files_gt[ti])
                    val_H = np.asarray(tmp).astype(np.float32)  # HxWxC

                    # Load LR image
                    tmp = _load_img_array(files_lr[ti])
                    val_L = np.asarray(tmp).astype(np.float32)  # HxWxC
                    val_L = np.transpose(val_L, [2, 0, 1])      # CxHxW
                    val_L = val_L[np.newaxis, ...]            # BxCxHxW

                    val_L = Variable(torch.from_numpy(val_L.copy()), volatile=True).cuda()

                    batch_S = model_G(val_L)
                    # Output 
                    image_out = (batch_S).cpu().data.numpy()
                    image_out = np.transpose(image_out[0], [1, 2, 0])  # HxWxC
                    image_out = np.clip(image_out, 0. , 1.)      # CxHxW
                    
                    # Save to file
                    image_out = ((image_out)*255).astype(np.uint8)
                    # Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

                    # PSNR on Y channel
                    img_gt = (val_H*255).astype(np.uint8)
                    CROP_S = 4
                    psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

            if psnr_best < np.mean(np.asarray(psnrs)):
                SaveCheckpoint(best=True)
                psnr_best=np.mean(np.asarray(psnrs))

            print('AVG PSNR: Validation: {}, best:{}'.format(np.mean(np.asarray(psnrs)),psnr_best))
                
            writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
            writer.flush()
