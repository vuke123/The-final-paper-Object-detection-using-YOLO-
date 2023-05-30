import torch 
import os 
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform = None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C 
    
    def __len__(self): 
        return len(self.annotations)
    
    def __getitem__(self, index): 
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        boxes = []
        with open(label_path) as f: 
            for label in f.readlines(): 
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split() 
                ]

                boxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        
        for box in boxes: 
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x) 
            x_cell, y_cell = self.S*x - j, self.S * y - i
            width_cell, height_cell = ( 
                width * self.S,
                height * self.S 
            )   

            if label_matrix[i,j,20] == 0: 
                label_matrix[i,j,20] = 1 
                box_coordinates = torch.tensor( 
                    [x_cell, y_cell ,width_cell ,height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
        return image, label_matrix
    
     
class VOCDataset2(torch.utils.data.Dataset): 
    def __init__(
            self, csv_file, deep_neural, img_dir, images_height=448, images_width=448, S=7, B=2, C=5, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform 
        self.S = S 
        self.B = B
        self.C = C
        self.deep_neural = deep_neural
        self.images_height = images_height
        self.images_width = images_width

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index): 

        boxes = []

        image_name = self.annotations.loc[index, 'frame']
        xmin_values = self.annotations.loc[index, 'xmin']
        xmin_values = xmin_values.replace('[', '').replace(']', '')
        xmin_values = xmin_values.split(',')
        xmax_values = self.annotations.loc[index, 'xmax']
        xmax_values = xmax_values.replace('[', '').replace(']', '')
        xmax_values = xmax_values.split(',')
        ymax_values = self.annotations.loc[index, 'ymax']
        ymax_values = ymax_values.replace('[', '').replace(']', '')
        ymax_values = ymax_values.split(',')
        ymin_values = self.annotations.loc[index, 'ymin']
        ymin_values = ymin_values.replace('[', '').replace(']', '')
        ymin_values = ymin_values.split(',')
        class_object_values = self.annotations.loc[index, 'class_id']
        class_object_values = class_object_values.replace('[', '').replace(']', '')
        class_object_values = class_object_values.split(',')
        
        length_values = len(xmin_values)
        
        if self.deep_neural == "yolo":
            resized_metric = 448
        elif self.deep_neural == "resnet":
            resized_metric = 224

        for i in range(length_values):
            xmin = float(xmin_values[i])
            xmax = float(xmax_values[i])
            ymin = float(ymin_values[i])
            ymax = float(ymax_values[i])
            
            if self.images_width != resized_metric or self.images_height != resized_metric: 
               xmin = xmin / (self.images_width / resized_metric)   #assume we use resnet where images are resized 224x224 than 224 
               xmax = xmax / (self.images_width / resized_metric) 
               ymin = ymin / (self.images_height / resized_metric)
               ymax = ymax / (self.images_height/ resized_metric)
            
            class_object = int(class_object_values[i]) - 1

            x = (xmin + xmax - 0.001) / (resized_metric * 2) #(/2/224), we want to get size and coordinates in percentage
            y = (ymin + ymax - 0.001) / (resized_metric * 2)
            w = (xmax - xmin) / (resized_metric * 2)
            h = (ymax - ymin) / (resized_metric * 2)

            i, j = int(self.S * y), int(self.S * x) 

            index = index + 1
            boxes.append([class_object, x, y, w, h])
            if (x > 1):
                print()
            
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path)        
        boxes = torch.tensor(boxes)
        
        #transform
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        
        for box in boxes: 
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x) 
            x_cell, y_cell = self.S*x - j, self.S * y - i
            width_cell, height_cell = ( 
                width * self.S,
                height * self.S 
            )   

            if label_matrix[i,j,5] == 0: 
                label_matrix[i,j,5] = 1 
                box_coordinates = torch.tensor( 
                    [x_cell, y_cell ,width_cell ,height_cell]
                )
                label_matrix[i, j, 6:10] = box_coordinates
                label_matrix[i, j, class_label] = 1
            

        return image, label_matrix