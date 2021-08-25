import torch
from torch import nn
from torchvision.models import vgg
from pprint import pprint
import torch.nn.functional as F

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):
    batch_sz = confGT_batch.size()[0]
    indices = torch.argmax(confGT_batch, dim=1)
    
    orientGT_batch = orientGT_batch[torch.arange(0, batch_sz), indices]
    orient_batch = orient_batch[torch.arange(0, batch_sz), indices]

    theta_diff = torch.atan2(orientGT_batch[:, 1], orientGT_batch[:, 0])
    estimated_theta_diff = torch.atan2(orient_batch[:, 1], orient_batch[:, 0])

    return - 1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class Deep3DBox(nn.Module):
    def __init__(self, bins = 2,
                       w = 0.4, 
                       pretrained_vgg=True,
                       ):
        super().__init__()
        self.bins = bins
        self.features = vgg.vgg19_bn(pretrained = pretrained_vgg).features
        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins * 2)
        )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins)
        )
        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 3)
        )


    def forward(self, x):
        x = self.features(x) # bs, 512, 7, 7
        x = x.view(-1, 512 * 7 * 7)
        orientation = self.orientation(x) # bs, 4
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=-1)
        confidence = self.confidence(x) # bs, 2
        dimension = self.dimension(x) # bs, 3
        return orientation, confidence, dimension

if __name__ == '__main__':
    model = Deep3DBox()
    # pprint(model.features)
    x = torch.rand([1,3,224,224])
    orientation, confidence, dimension = model(x)
    print(orientation.shape)
    print(confidence.shape)
    print(dimension.shape)