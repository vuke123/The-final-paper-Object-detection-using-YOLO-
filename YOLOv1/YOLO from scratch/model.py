import torch 
import torch.nn as nn

architecture = [
(7,64, 2, 3), 
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
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.conv(x))
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): 
        super(Yolov1, self).__init__()
        self.architecture = architecture
        self.in_channels = in_channels
        self.darknet = self.Create_CNNLayers(self.architecture)
        self.fcs = self.Create_FCLayers(**kwargs)

    def forward(self, x): 
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
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
    #nn creates model, *layers unpack list

    def Create_FCLayers(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 512), #4096 instead but takes more time
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S*S*(C+B*5)), # 7x7x30
        )

def test(S=7, B=2, C=20): #data from paper
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((1,3, 448,448))
    print(model(x).shape)

test() 