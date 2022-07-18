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


from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter

EXP_NAME = "SP-LUT"

UPSCALE = 4     # upscaling factor

NB_BATCH = 32        # mini-batch
CROP_SIZE = 48       # input LR training patch size

START_ITER = 0      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER = 2000000    # Total number of training iterations

I_DISPLAY = 200     # display info every N iteration
I_VALIDATION = 200  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './val/'      # Validation images

LR_G = 1e-3         # Learning rate for the generator

QQ=2**4
OUT_NUM=4
VERSION = "SPLUT_M_{}_{}".format(LR_G,OUT_NUM)

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

### A lightweight deep network ###
class SRNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(SRNet, self).__init__()

        self.upscale = upscale
        self.lvls = 16
        self.quant= BASEQ(self.lvls ,8.0)
        self.out_channal= OUT_NUM

        #cwh
        self.lut122=nn.Sequential(
            nn.Conv2d(1, 64, [2,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut221=nn.Sequential(
            nn.Conv2d(2,  64, [2,1], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut212=nn.Sequential(
            nn.Conv2d(2,  64, [1,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 8, 1, stride=1, padding=0, dilation=1)
        )

        self.lut221_c12=nn.Sequential(
            nn.Conv2d(2,  64, [2,1], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        )

        self.lut212_c34=nn.Sequential(
            nn.Conv2d(2,  64, [1,2], stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1),
            nn.GELU(),
            nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)
        x = self.lut122(x_in)+x_in[:,:,:H-1,:W-1]
        x_temp=x_in
        x221 = self.lut221(self.quant(F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect')))
        x212 = self.lut212(self.quant(F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect')))
        x=(x221+x212)/2.0+x

        x2 = self.lut221_c12(self.quant(F.pad(x[:,0:2,:,:],(0,0,0,1), mode='reflect')+F.pad(x[:,2:4,:,:],(0,0,1,0), mode='reflect')))
        x3 = self.lut212_c34(self.quant(F.pad(x[:,4:6,:,:],(0,1,0,0), mode='reflect')+F.pad(x[:,6: ,:,:],(1,0,0,0), mode='reflect')))
        x=(x2+x3)/2.0+x_temp[:,:,:H-1,:W-1]
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale*(H-1), self.upscale*(W-1))
        return x

if __name__ == '__main__':

    model_G_A = SRNet(upscale=UPSCALE).cuda()
    model_G_B = SRNet(upscale=UPSCALE).cuda()

    ## Optimizers
    params_G_A = list(filter(lambda p: p.requires_grad, model_G_A.parameters()))
    params_G_B = list(filter(lambda p: p.requires_grad, model_G_B.parameters()))
    opt_G = optim.Adam([{'params':params_G_A},{'params':params_G_B}],lr=LR_G)

    ## Load saved params
    if START_ITER > 0:
        lm = torch.load('checkpoint/{}/model_G_A_i{:06d}.pth'.format(str(VERSION), START_ITER))
        model_G_A.load_state_dict(lm.state_dict(), strict=True)

        lm = torch.load('checkpoint/{}/model_G_B_i{:06d}.pth'.format(str(VERSION), START_ITER))
        model_G_B.load_state_dict(lm.state_dict(), strict=True)

        lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
        opt_G.load_state_dict(lm.state_dict())

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

    def SaveCheckpoint(i, best=False):
        str_best = ''
        if best:
            str_best = '_best'

        torch.save(model_G_A, 'checkpoint/{}/model_G_A_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
        torch.save(model_G_B, 'checkpoint/{}/model_G_B_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
        torch.save(opt_G, 'checkpoint/{}/opt_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best))
        print("Checkpoint saved")

    def SpaceInference(model,batch):
        batch_S1 = model(F.pad(batch, (0,1,0,1), mode='reflect'))
        batch_S = torch.clamp(batch_S1,-1,1)
        return batch_S

    ### TRAINING
    for i in tqdm(range(START_ITER+1, NB_ITER+1)):

        model_G_A.train()
        model_G_B.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H,batch_Stage1 = Iter_H.dequeue()
        
        batch_H = Variable(torch.from_numpy(batch_H)).cuda()      # BxCxHxW, range [0,1]
        batch_Stage1 = Variable(torch.from_numpy(batch_Stage1)).cuda()      # BxCxHxW, range [0,1]
        batch_L = Variable(torch.from_numpy(batch_L)).cuda()      # BxCxHxW, range [0,1]

        #[0,1]-->[0,255]
        batch_L225 = torch.floor(batch_L*255)
        batch_L_A= batch_L225// QQ
        batch_L_B= batch_L225 % QQ
        batch_L_A=batch_L_A/QQ
        batch_L_B=batch_L_B/QQ

        dT += time.time() - st

        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        batch_S_A=SpaceInference(model_G_A,batch_L_A)
        batch_S_B=SpaceInference(model_G_B,batch_L_B)
        batch_S_all = torch.clamp(batch_S_A+batch_S_B,0,1)

        loss_Pixel = torch.mean( ((batch_S_all - batch_H)**2))
        loss_G = loss_Pixel

        # Update
        loss_G.backward()
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
            SaveCheckpoint(i)

        ## Validation
        if i % I_VALIDATION == 0:
            with torch.no_grad():
                model_G_A.eval()
                model_G_B.eval()

                # Test for validation images
                files_gt = glob.glob(VAL_DIR + '/HR/*.png')
                files_gt.sort()
                files_lr = glob.glob(VAL_DIR + '/LR/*.png')
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
                    #pdb.set_trace()
                    val_L = torch.floor(val_L*255)
                    val_L_A= val_L// QQ
                    val_L_B= val_L % QQ
                    val_L_A=val_L_A/QQ
                    val_L_B=val_L_B/QQ

                    batch_S_A=SpaceInference(model_G_A,val_L_A)
                    batch_S_B=SpaceInference(model_G_B,val_L_B)
                    batch_S = batch_S_A+batch_S_B

                    # Output 
                    image_out = (batch_S).cpu().data.numpy()
                    image_out = np.transpose(image_out[0], [1, 2, 0])  # HxWxC
                    image_out = np.clip(image_out, 0. , 1.)      # CxHxW
                    
                    # Save to file
                    # image_out = ((image_out)*255).astype(np.uint8)
                    # Image.fromarray(image_out).save('result/{}/{}.png'.format(str(VERSION), fn.split('/')[-1]))

                    # PSNR on Y channel
                    img_gt = (val_H*255).astype(np.uint8)
                    CROP_S = 4
                    psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

            print('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))

            writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
            writer.flush()
