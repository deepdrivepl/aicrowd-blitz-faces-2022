import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import torchvision

import transforms as T
from engine import train_one_epoch, evaluate
import utils
import albumentations as A

    
class TestMaskDataset(torch.utils.data.Dataset):
    def __init__(self, root, class_df, transforms):
        self.root = root
        self.transforms = transforms
        self.class_df = class_df
    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, (self.class_df["ImageID"][idx]+".jpg"))
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.class_df.shape[0]
    

if __name__ == '__main__':
    test_images = 'data/test'
    
    classes = ["cloth", "KN95","N95", "surgical"]
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=5)
    model.load_state_dict(torch.load('model_weights.pth'))
    
    test_df = pd.read_csv("data/sample_submission.csv")
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    test_ds = TestMaskDataset(test_images, test_df, test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=8)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    labels = []
    boxes = []
    
    for idx, img in enumerate(test_loader):
        img = img.to(device)
        res = model(img)
        if res[0]["scores"].shape[0] >= 1:
            box = res[0]['boxes'].cpu().detach().numpy()[0]
#             box = (box.astype(int)).tolist()
            box = box.tolist()
            label = int((res[0]["labels"].cpu().detach().numpy()[0]))
        else:
            label = ""
            box = [0, 0, 0, 0]
        
        labels.append(label)
        boxes.append(box)
    newlist = [classes[label-1] if isinstance(label, int) else label for label in labels]
    test_df["bbox"] = boxes
    test_df["masktype"] = newlist
    
    test_df.to_csv("submission.csv", index=False)
    