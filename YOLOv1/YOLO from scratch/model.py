import torch 
import torch.nn as nn


device="cpu"
if torch.cuda.is_available():
    device="cuda"
print(device)

architecture = [
(7,64, 2, 3), #kernel, output channels, different kernels, in cn layer, stride, padding
"MaxPool",
(3, 192, 1, 1),
"MaxPool",
(1,128,1,0),
(3,256,1,1),
(1,256,1,0),
(3,512,1,1),
"MaxPool",
(1,256,1,0), 
(3,512,1,1),
(1,256,1,0), 
(3,512,1,1),
(1,256,1,0), 
(3,512,1,1),
(1,256,1,0), 
(3,512,1,1),
(1,512,1,0),
(3,1024,1,1),
"MaxPool",
(1,512,1,0), (3,1024, 1, 1),
(1,512,1,0), (3,1024, 1, 1),
(1,512,1,0), (3,1024, 1, 1),
(1,512,1,0), (3,1024, 1, 1),
(3,1024,1,1),
(3,1024,2,1),
(3,1024,1,1),
(3,1024,1,1),
]

class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): 
        super(Yolov1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self.Create_CNNLayers(self.architecture)
        self.fcs = self.Create_FCLayers(**kwargs)
        self.to(device) # Move the entire model to CUDA if possible

    def forward(self, x): 
        
        x = self.darknet(x) 
        flattenx = torch.flatten(x, start_dim=1)
        return self.fcs(flattenx)  #start dim = 1 because we do not want to flatten
    #number of examples 

    def Create_CNNLayers(self, architecture): 
        layers = []
        in_channels = self.in_channels
        
        for x in architecture: 
            if type(x) == tuple:
                layers += [CNNLayer(
                    in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                ]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers) 
    #nn creates model, *layers --nn.seq --> unpack list

    def Create_FCLayers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 512), #1024*S*S is in the paper input features
            #4096 is output instead but takes more time
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S*S*(C+B*5)), # 7x7x30 || 7x7x15 
        )



