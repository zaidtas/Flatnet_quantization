*** Work in progress


Please run main.py for training from scratch
Each parser argument in the main.py file has a corresponding instruction in the form of comment.

**Needs data in the form of filenames mentioned in the parser arguments of main.py
**Currently filenames are for Imagenet measurements used for ICCV 2019


***REGARDING INITIALIZATIONS***

Transpose Initializations:
'flatcam_prototype2_calibdata.mat' contains the calibration matrices : Phi_L and Phi_R. They are named as P1 and Q1 respectively once you load the mat file. Please note that there are separate P1 and Q1 for each channel (b,gr,gb,r). For the paper, we use only one of them (P1b and Q1b) for initializing the weights (W_1 and W_2) of trainable inversion layer.


Random Toeplitz Initializations:
'phil_toep_slope22.mat' and 'phir_toep_slope22' contain the random toeplitz matrices corresponding to W_1 and W_2 of the trainable inversion layer. 
