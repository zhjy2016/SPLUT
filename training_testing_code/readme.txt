## Dependency

- Python 3.6
- PyTorch 
- glob
- numpy
- pillow
- tqdm
- tensorboardx
- PIL
- basicsr

## Code

- `Train_SPLUT_L/M/S.py`: the code for training SPLUT
- `Transfer_SPLUT_L/M/S.py`: the code for transferring mapping modules to LUTs
- `Inference_SPLUT_L/M/S.py`: the code for testing SPLUT

## Training

1. Prepare folders of `./log` and `./transfer`
2. Prepare DIV2K training images in `./train`

- [HR images]: `./train/DIV2K_train_HR/*.png`
- [LR images]:`./train/DIV2K_train_LR_bicubic/X4/*.png`

3. Prepare LR/HR images  in `./val`

4. Run `python Train_SPLUT_L/M/S.py`

5. Checkpoints used in our paper have been saved in `./checkpoint`

- Training log will be generated in `./log`. 

## Transferring

1. Run `python Transfer_SPLUT_L/M/S.py`

2. The resulting LUT will be saved in the folder of `./transfer/`

## Inference using LUT

1. The Set5 LR/HR images for testing are already included in `./test_dataset`. The users can also evaluate other images.
2. Run `python Inference_SPLUT_L/M/S.py`

