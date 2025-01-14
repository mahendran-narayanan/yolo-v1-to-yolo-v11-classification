import torch
from torch.nn import Conv2d,MaxPool2d

#Yolov1

model = torch.nn.Sequential(
Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2)),
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1)),
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1)),
Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1)),
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1)),
MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# Extend
)

print(model)