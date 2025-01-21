import torch
from torch.nn import Conv2d,MaxPool2d,BatchNorm2d,Flatten,Linear

class ConvModule(torch.nn.Module):
	def __init__(self,inp,out,kernels,strides,mpk,mps):
		super(ConvModule, self).__init__()
		self.conv = Conv2d(inp, out, kernel_size=(kernels, kernels), stride=(strides, strides))
		self.mp = MaxPool2d(mpk,mps)
	def forward(self,x):
		x = self.conv(x)
		x = self.mp(x)
		return x

class TwoConv(torch.nn.Module):
	def __init__(self,inp,mid,out,kernels,strides,k2,s2):
		super(TwoConv, self).__init__()
		self.conv = Conv2d(inp, mid, kernel_size=(kernels, kernels), stride=(strides, strides))
		self.conv2 = Conv2d(mid, out, kernel_size=(k2, k2), stride=(s2, s2))
	def forward(self,x):
		x = self.conv(x)
		x = self.conv2(x)
		return x

class Yolov1(torch.nn.Module):
	"""
	Paper Title: You Only Look Once: Unified, Real-Time Object Detection
	Paper Link: https://arxiv.org/abs/1506.02640
	Proposed model is modified to be used as classification model for Imagenet 1K classes.
	"""
	def __init__(self):
		super(Yolov1, self).__init__()
		self.conv1 = ConvModule(3, 64, 7,2,2,2)
		self.conv2 = ConvModule(64,192, 3,1,2,2)
		self.conv3 = Conv2d(192, 128, kernel_size=(1,1))
		self.conv4 = Conv2d(128, 256, kernel_size=(3,3), stride=(1,1))
		self.conv5 = Conv2d(256, 256, kernel_size=(1,1), stride=(1,1))
		self.conv6 = Conv2d(256, 512, kernel_size=(3,3), stride=(1,1))
		self.maxp1 = MaxPool2d(2,2)
		self.conv7 = torch.nn.ModuleList(
				TwoConv(512,256, 512, 1,1,3,1) for _ in range(4)
				)
		self.conv8 = Conv2d(512, 512, kernel_size=(1,1), stride=(1,1))
		self.conv9 = Conv2d(512, 1024, kernel_size=(3,3), stride=(1,1))
		self.maxp2 = MaxPool2d(2,2)
		self.conv10 = torch.nn.ModuleList(
				TwoConv(1024,512, 1024, 1,1,3,1) for _ in range(2)
				)
		self.conv11 = Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1))
		self.conv12 = Conv2d(1024, 1024, kernel_size=(3,3), stride=(2,2))
		self.conv13 = Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1))
		self.conv14 = Conv2d(1024, 1024, kernel_size=(3,3), stride=(1,1))
		self.flat = Flatten()
		self.dense1 = Linear(1024,4096)
		self.dense2 = Linear(4096,1000)
	def forward(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.maxp1(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = self.conv9(x)
		x = self.maxp2(x)
		x = self.conv10(x)
		x = self.conv11(x)
		x = self.conv12(x)
		x = self.conv13(x)
		x = self.conv14(x)
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dense2(x)
		return x

model = Yolov1()