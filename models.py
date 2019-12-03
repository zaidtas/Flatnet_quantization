import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x

	
	
class double_conv2(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv2, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x    

	
	

class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			double_conv2(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=False):
		super(up, self).__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffX = x1.size()[2] - x2.size()[2]
		diffY = x1.size()[3] - x2.size()[3]
		x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
						diffY // 2, int(diffY / 2)))
		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=1)

	def forward(self, x):
		x = self.conv(x)
		return x


# In[7]:


def swish(x):
	return x * torch.sigmoid(x)


# In[8]:


class FlatNet(nn.Module):
	def __init__(self, phil,phir,n_channels=4):
		super(FlatNet, self).__init__()
		self.inc = inconv(n_channels, 128)
		self.down1 = down(128, 256)
		self.down2 = down(256, 512)
		self.down3 = down(512, 1024)
		self.down4 = down(1024, 1024)
		self.up1 = up(2048, 512)
		self.up2 = up(1024, 256)
		self.up3 = up(512, 128)
		self.up4 = up(256, 128)
		self.outc = outconv(128, 3)
		self.PhiL =nn.Parameter(torch.tensor(phil)) 
		self.PhiR=nn.Parameter(torch.tensor(phir)) 
		self.bn=nn.BatchNorm2d(4,momentum=0.99)
	def forward(self, Xinp):
		
		X0=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,0,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X11=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,1,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X12=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,2,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X2=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,3,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		Xout=torch.cat((X2,X12,X11,X0),3)
		x = Xout.permute(0,3,1,2)

		x = self.bn(x)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		# print(x.shape)

			
		return torch.sigmoid(x)




class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		#self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

		self.conv2 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		#self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		#self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		#self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		#self.bn5 = nn.BatchNorm2d(256)
		self.conv6 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
		self.bn6 = nn.BatchNorm2d(128)
		#self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		#self.bn7 = nn.BatchNorm2d(512)
		#self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
		#self.bn8 = nn.BatchNorm2d(512)

		# Replaced original paper FC layers with FCN
		self.conv9 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

	def forward(self, x):
		#x = swish(self.conv1(x))

		x = swish(self.bn2(self.conv2(x)))
		#x = swish(self.bn3(self.conv3(x)))
		x = swish(self.bn4(self.conv4(x)))
		#x = swish(self.bn5(self.conv5(x)))
		x = swish(self.bn6(self.conv6(x)))
		#x = swish(self.bn7(self.conv7(x)))
		#x = swish(self.bn8(self.conv8(x)))
		#print(x.shape)
		x = self.conv9(x)
		return F.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

