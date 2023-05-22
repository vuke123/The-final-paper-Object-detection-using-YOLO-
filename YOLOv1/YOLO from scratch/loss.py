import torch 
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20): 
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        #predictions_flatten = torch.flatten(predictions, end_dim=-1)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        
        exists_box = target[..., 20].unsqueeze(3) #adds new dimension of size 1

        ##box coordinates and size(w&h) -------------------
        box_predictions = exists_box * (
            (
            bestbox * predictions[..., 26:30]
            + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[...,21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[...,2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        #(N, S, S, 4) - (N*S*S, 4)
        ##box loss ----------------
        box_loss = self.mse( 
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )


        ##object loss -------------------
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * 
            predictions[..., 20:21]
        )

        #(N*S*S,1)
        object_loss = self.mse( 
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[...,20:21])
        )

        ##no object loss -------------------
        no_object_loss = self.mse( 
            torch.flatten((1-exists_box) * predictions[...,20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
            #if 10 is first dimension size it will be [10, 49] because 
            # we did not flatten first dimension 
        )
        #flatten with start_dim=1 is not official, could be different

        no_object_loss += self.mse( 
            torch.flatten((1-exists_box) * predictions[...,25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        ##class loss -------------------

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., 0:20], end_dim=-2),
            #[10*7*7, 20]
            #[490, 20] group of classes-wise, not element-wise [490*20]
            torch.flatten(exists_box * target[..., :20], end_dim = -2),
        ) 

        loss = (
            self.lambda_coord * box_loss
            + object_loss 
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss 
                 
                           
