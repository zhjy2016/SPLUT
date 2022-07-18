from os import mkdir
from os.path import isdir
import torch
import torch.nn.functional as F
import numpy as np

from Train_SPLUT_S import _baseq,BASEQ,SRNet,UPSCALE,VERSION

MODEL_PATH = "./checkpoint/{}".format(VERSION)   # Trained SR net params
ITER=318000

LUT_NUM_LIST=[1,2,2,3,3]
KERNAL_LIST=[122,221,212,221,212]
img_bits=4

class SRNet_LUT(SRNet):
    def __init__(self, upscale=4):
        super(SRNet_LUT, self).__init__()

    def forward(self, x_in):
        if LUT_NUM==1:
            x_out = self.lut122(x_in)
        elif LUT_NUM==2 and KERNAL==221:
            x_out = self.lut221(x_in)
        elif LUT_NUM==2 and KERNAL==212:
            x_out = self.lut212(x_in)
        elif LUT_NUM==3 and KERNAL==221:
            x_out = self.lut221_c12(x_in)
        elif LUT_NUM==3 and KERNAL==212:
            x_out = self.lut212_c34(x_in)
        return x_out

if __name__ == '__main__':

    model_G_A = SRNet_LUT(upscale=UPSCALE)
    model_G_B = SRNet_LUT(upscale=UPSCALE)
    lm = torch.load('{}/model_G_A_i{:06d}.pth'.format(MODEL_PATH, ITER))
    model_G_A.load_state_dict(lm.state_dict(), strict=True)
    lm = torch.load('{}/model_G_B_i{:06d}.pth'.format(MODEL_PATH, ITER))
    model_G_B.load_state_dict(lm.state_dict(), strict=True)

    ### Extract input-output pairs
    with torch.no_grad():
        model_G_A.eval()
        model_G_B.eval()
        for ind in range(len(LUT_NUM_LIST)):
            LUT_NUM=LUT_NUM_LIST[ind]
            KERNAL=KERNAL_LIST[ind]

            if LUT_NUM==1:
                L = 2 ** img_bits
                base_step_ind=torch.arange(0, L, 1)

                base_steps=16
                base=base_steps*base_step_ind/255.0
                index_4D=torch.meshgrid(base,base,base,base)
                onebyfourth=torch.cat([index_4D[0].flatten().unsqueeze(1),index_4D[1].flatten().unsqueeze(1),index_4D[2].flatten().unsqueeze(1),index_4D[3].flatten().unsqueeze(1)],1)
                
                base_steps_B= 1
                base_B=base_steps_B*base_step_ind/255.0
                index_4D_B=torch.meshgrid(base_B,base_B,base_B,base_B)
                onebyfourth_B=torch.cat([index_4D_B[0].flatten().unsqueeze(1),index_4D_B[1].flatten().unsqueeze(1),index_4D_B[2].flatten().unsqueeze(1),index_4D_B[3].flatten().unsqueeze(1)],1)
                
            else:
                a_rang = 8.0
                L = model_G_A.lvls
                base_steps=2* a_rang/L
                base_step_ind=torch.arange(0, L, 1)-0.5*L
                base=base_steps*base_step_ind
                index_4D=torch.meshgrid(base,base,base,base)
                onebyfourth=torch.cat([index_4D[0].flatten().unsqueeze(1),index_4D[1].flatten().unsqueeze(1),index_4D[2].flatten().unsqueeze(1),index_4D[3].flatten().unsqueeze(1)],1)

            if LUT_NUM==1:
                input_tensor   = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2)
                input_tensor_B = onebyfourth_B.unsqueeze(1).unsqueeze(1).reshape(-1,1,2,2)
            elif KERNAL==221:
                input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,2,2,1)
                input_tensor_B = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,2,2,1)
            elif KERNAL==212:
                input_tensor = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,2,1,2)
                input_tensor_B = onebyfourth.unsqueeze(1).unsqueeze(1).reshape(-1,2,1,2)
            print("Input size: ", input_tensor.size())

            # Split input to not over GPU memory
            B = input_tensor.size(0) // 100
            outputs_A = []
            outputs_B = []

            for b in range(100):
                if b == 99:
                    batch_output_A = model_G_A(input_tensor[b*B:])
                    batch_output_B = model_G_B(input_tensor_B[b*B:])
                else:
                    batch_output_A = model_G_A(input_tensor[b*B:(b+1)*B])
                    batch_output_B = model_G_B(input_tensor_B[b*B:(b+1)*B])

                outputs_A += [ batch_output_A ]
                outputs_B += [ batch_output_B ]
            
            results_A = np.concatenate(outputs_A, 0)
            results_B = np.concatenate(outputs_B, 0)
            print("Resulting LUT size: ", results_A.shape)
            if not isdir('transfer/{}'.format(str(VERSION))):
                mkdir('transfer/{}'.format(str(VERSION)))
            np.save("./transfer/{}/LUT{}_K{}_Model_S_A".format(VERSION,LUT_NUM,KERNAL), results_A)
            np.save("./transfer/{}/LUT{}_K{}_Model_S_B".format(VERSION,LUT_NUM,KERNAL), results_B)