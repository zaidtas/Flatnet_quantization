import torch
import scipy.io as sio
import numpy as np
import os
from skimage.color import rgb2gray
import skimage.io
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as rsz
import torch.optim as optim
import os
from torch_vgg import Vgg16
from models import*
from fns_all import*
from dataloader import*
import argparse
from torch.utils import data
import torchvision.transforms as transforms
import skimage.transform
import copy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()
#model and data locs
parser.add_argument('--train_meas_filenames', default='filenames/train_meas_ilsvrc_flatcam.txt') #train measurement filenames
parser.add_argument('--val_meas_filenames', default='filenames/val_meas_ilsvrc_flatcam_smaller.txt') #validation measurement filenames
parser.add_argument('--train_orig_filenames', default='filenames/train_orig_ilsvrc_flatcam.txt') #train ground truth filenames
parser.add_argument('--val_orig_filenames', default='filenames/val_orig_ilsvrc_flatcam_smaller.txt') #validation ground truth filenames
parser.add_argument('--modelRoot', default='flatnet') #set it to desired save directory for each learned model
parser.add_argument('--checkpoint', default='') #Provide this while loading from a pretrained model
parser.add_argument('--wtp', default=1.2, type=float) #Weight for perceptual vgg loss
parser.add_argument('--wtmse', default=1, type=float) #weight for MSE loss
parser.add_argument('--wta', default=0.6, type=float) #weight for adversarial loss
parser.add_argument('--generatorLR', default=1e-4, type=float) #initial learning rate for generator
parser.add_argument('--discriminatorLR', default=1e-4, type=float) #initial learning rate for discriminator
parser.add_argument('--dev_id', default=3, type=int) #set the gpu id to use
parser.add_argument('--init', default='Transpose') #Set 'Transpose' for transpose initialization or 'Random' for initialization
parser.add_argument('--numEpoch', default=20,type=int) #Total number of epochs you want your network to run for
parser.add_argument('--disPreEpochs', default=5,type=int) #Number epochs for discriminator pretraining
parser.add_argument('--pretrain',dest='pretrain', action='store_true') #While training for scratch, its wise to pretrain the discriminator
parser.set_defaults(pretrain=True)

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.dev_id)
device = torch.device("cuda")

data = '/root/data/Amplitude Mask/models/' ##where do you want to save your saved model
savedir = os.path.join(data, opt.modelRoot)

if not os.path.exists(savedir):
	os.mkdir(savedir)

batchsize = 4
vla = float('inf')
k = 0
val_err = []
train_err = []

if opt.init=='Transpose':
	print('Loading calibrated files')
	d=sio.loadmat('flatcam_prototype2_calibdata.mat') ##Initialize the weight matrices with transpose
	phil=np.zeros((500,256,1))
	phir=np.zeros((620,256,1))
	phil[:,:,0]=d['P1gb']
	phir[:,:,0]=d['Q1gb']
	phil=phil.astype('float32')
	phir=phir.astype('float32')
else:
	print('Loading Random Toeplitz') ##Initialize the weight matrices with random toeplitz-like matrices
	phil=np.zeros((500,256,1))
	phir=np.zeros((620,256,1))
	pl = sio.loadmat('phil_toep_slope22.mat')
	pr = sio.loadmat('phir_toep_slope22.mat')
	phil[:,:,0] = pl['phil'][:,:,0]
	phir[:,:,0] = pr['phir'][:,:,0]
	phil=phil.astype('float32')
	phir=phir.astype('float32')


gen = FlatNet(phil,phir,4).to(device)
vgg = Vgg16(requires_grad=False).to(device)
dis = Discriminator().to(device)

gen_criterion = nn.MSELoss()
dis_criterion = nn.BCELoss()

ei = 0
train_error = []
val_error = []

optim_gen = torch.optim.Adam(gen.parameters(), lr= opt.generatorLR)
optim_dis = torch.optim.Adam(dis.parameters(), lr= opt.discriminatorLR)
vla = float('inf')
if opt.checkpoint:
	checkpoint = os.path.join(data, opt.checkpoint)
	ckpt = torch.load(checkpoint+'/latest.tar')
	optim_gen.load_state_dict(ckpt['optimizerG_state_dict'])
	optim_dis.load_state_dict(ckpt['optimizerD_state_dict'])
	dis.load_state_dict(ckpt['dis_state_dict'])
	gen.load_state_dict(ckpt['gen_state_dict'])
	ei = ckpt['last_finished_epoch'] + 1
	val_error = ckpt['val_err']
	train_error = ckpt['train_err']
	vla = min(ckpt['val_err'])
	print('Loaded checkpoint from:'+checkpoint+'/latest.tar')

params_train = {'batch_size': 4,
		  'shuffle': True,
		  'num_workers': 4}

params_val = {'batch_size': 1,
		  'shuffle': False,
		  'num_workers': 4}
train_loader = torch.utils.data.DataLoader(DatasetFromFilenames(opt.train_meas_filenames,opt.train_orig_filenames), **params_train)
val_loader = torch.utils.data.DataLoader(DatasetFromFilenames(opt.val_meas_filenames,opt.val_orig_filenames), **params_val)


wts = [opt.wtmse, opt.wtp, opt.wta]

disc_err = []
if opt.pretrain and not opt.checkpoint:
	disc_err = train_discriminator_epoch(gen, dis, optim_dis, dis_criterion, train_loader, opt.disPreEpochs, disc_err, device)
torch.save(dis.state_dict(), savedir+'/pretrained_disc.tar')



for e in range(ei,opt.numEpoch):
	train_error, val_error, disc_err, vla, Xvalout = train_full_epoch(gen, dis, vgg, wts, optim_gen, optim_dis, train_loader, val_loader, gen_criterion, dis_criterion, device, vla, e, savedir, train_error, val_error, disc_err)
	Xvalout = Xvalout.cpu()
	ims = Xvalout.detach().numpy()
	ims = ims[0, :, :, :]
	ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
	ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
	skimage.io.imsave(savedir+'/latest.png', ims)

	dict_save = {
			'gen_state_dict': gen.state_dict(),
			'dis_state_dict': dis.state_dict(),
			'optimizerG_state_dict': optim_gen.state_dict(),
			'optimizerD_state_dict': optim_dis.state_dict(),
			'train_err': train_error,
			'val_err': val_error,
			'disc_err': disc_err,
			'last_finished_epoch': e,
			'opt': opt,
			'vla': vla}
	torch.save(dict_save, savedir+'/latest.tar')
	savename = '/phil_epoch%d' % e
	np.save(savedir+savename, gen.PhiL.detach().cpu().numpy())
	savename = '/phir_epoch%d' % e
	np.save(savedir+savename, gen.PhiR.detach().cpu().numpy())
	if e%2 == 0:
		for param_group in optim_gen.param_groups:
			param_group['lr'] = param_group['lr']/2
		for param_group in optim_dis.param_groups:
			param_group['lr'] = param_group['lr']/2

	print('Saved latest')




	











