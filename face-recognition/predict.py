#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import timm # !pip install git+https://github.com/rwightman/pytorch-image-models
from tqdm import tqdm



import pytorch_lightning
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer




import torch
# from torch.nn import functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import collections

from glob import glob
import cv2
import random

import numpy as np

batch_size_freezed = 128
batch_size_unfreezed = 1
train_size=384
test_size=384

def random_erase(img):
    # img = np.array(pilimg)

    h,w,_ = img.shape

    xx = np.random.random(2)
    xmin = int(min(xx)*w)
    xmax = int(max(xx)*w)

    yy = np.random.random(2)
    ymin = int(min(yy)*h)
    ymax = int(max(yy)*h)

    # print(ymin,ymax,xmin,xmax)

    img[ymin:ymax,xmin:xmax,:]=0

    return Image.fromarray(img)



def get_target_face(face_no, target_image):
    a = face_no//10
    b = face_no - 10*a
    # Top-Left x, y corrdinates of the specific face 
    x, y = a*216, b*216

    target_face = target_image[x:x+216, y:y+216]

    return target_face

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform,erase=True):
        self.transform = transform
        self.path = path
        
        self.images = glob(self.path+'/*.jpg')
        self.counter=0
        self.erase=erase

    def __len__(self):
        return len(self.images)*100

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = idx//100
        face_in_img = idx - 100*img_id

        image100 = cv2.imread(self.images[img_id])
        image = get_target_face(face_in_img,image100)
        
                
        img_id2 = random.randint(0,len(self.images)-1)
        face_in_img2 = random.randint(0,99)

        image100_2 = cv2.imread(self.images[img_id2])
        image2 = get_target_face(face_in_img2,image100_2)
        
        if self.erase:
            image_a = random_erase(image)
            image_p = random_erase(image)
            image_n = random_erase(image2)
        else:
            image_a = Image.fromarray(image)
            image_p = Image.fromarray(image)
            image_n = Image.fromarray(image2)

        if self.transform is not None:
            image_a = self.transform(image_a)
            image_p = self.transform(image_p)
            image_n = self.transform(image_n)
            
        sample = {'anchor': image_a,
                  'pos': image_p,
                  'neg': image_n}

        return sample

    def get_counts(self):
        return collections.defaultdict(int, self.dataframe.age.value_counts().to_dict())








augmentations = A.Compose(
                [A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.5, rotate_limit=20, p=0.8),
                A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=.5),
                    A.PiecewiseAffine(p=0.5),
                ], p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.5),
                A.OneOf([A.Solarize(),
                         A.Superpixels(),
                         A.Posterize()]),
                A.RandomGamma(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.3),
                A.ImageCompression(quality_lower=20, quality_upper=90,p=0.5)
                ])

transform_train = transforms.Compose([transforms.Resize([train_size, train_size]),
                              transforms.Lambda(lambda img: np.array(img)),
                              transforms.Lambda(lambda img: augmentations(image=img)['image']),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                     ])

transform_val = transforms.Compose([
    transforms.Resize([test_size, test_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



# f = FaceDataset('data/target', transform=transform_train)
# plt.figure(figsize=(20,7))
# for _ in f:
#     for i,k in enumerate(_):
#         print(i,k,_[k].shape)
#         plt.subplot(1,3,i+1)
#         plt.title(k)
#         plt.axis('off')
#         plt.imshow(UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(_[k]).permute(1,2,0).numpy()[:,:,::-1])
#     break



def getModel(backbone, train_size):
    model = nn.ModuleList([timm.create_model(backbone, pretrained=True, num_classes=0),])
    in_features = model[0](torch.rand(1,3,train_size, train_size)).shape[-1]

    model.append(nn.Sequential(torch.nn.LayerNorm([in_features, ]),
                          nn.Dropout(0.5),
                          nn.Linear(in_features, 128)))
    return model



class FaceRecognition(LightningModule):
    def __init__(self, net, train_size, test_size, transform_train, transform_val, batch_size=16, lr=1e-3,
                pt_fname=None, ckpt_fname=None):
        super().__init__()
        
        self.save_hyperparameters('batch_size', 'lr', 'train_size', 'test_size')

        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size

        self.example_input_array = torch.rand(1, 3, self.train_size, self.train_size)

        self.transform_train = transform_train
        self.transform_val = transform_val
        
        self.setup()
        
        self.loss=torch.nn.TripletMarginLoss()
        self.lr = lr
        self.net = net

    def custom_histogram_adder(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
    
    def freeze(self, epochs):
        for i,param in enumerate(self.net[0].parameters()):
            param.requires_grad = False
        for i,param in enumerate(self.net[1].parameters()):
            param.requires_grad = True

        self.epochs = epochs
    
    def unfreeze(self, epochs):
        for i,param in enumerate(self.net[0].parameters()):
            param.requires_grad = True
        for i,param in enumerate(self.net[1].parameters()):
            param.requires_grad = True
        self.epochs = epochs
        
    def freeze_bn(self):
        for module in self.modules():
            # print(module)
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                # print(module,'==> FREEZE!!!')
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def forward(self, x):
        x = self.net[0](x)
        return self.net[1](x)
    
    def on_fit_start(self):
        self.logger.log_hyperparams(self.hparams,
                                    metrics=#{})
        dict.fromkeys(['hp/val/loss'], float('NaN')))
    
    def _step(self, mode, batch, batch_idx):
        # print("batch_idx:",batch_idx)
        # for k in batch:
        #     print(k,batch[k].shape)
        anchor = self(batch['anchor'])
        pos = self(batch['pos'])
        neg = self(batch['neg'])

        # print('='*40)
        
        # triplet_loss(anchor, positive, negative)
        loss = self.loss(anchor, pos, neg)

        d = {}
                
        self.log(mode+'/loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        d['loss'] = loss

        return d
    
    def training_epoch_end(self, outputs):
        # logging histograms
        self.custom_histogram_adder()

    def validation_step_end(self, batch_parts):
        pass
#         # DP - size=0 problem https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validating-with-dataparallel
#         print('\n\nvalidation_step_end',type(batch_parts),len(batch_parts))
#         print('\n')

    def validation_epoch_end(self, outputs):
        valloss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('hp/val/loss', valloss)
        if self.current_epoch==0:
            self.plot_train_examples()
            self.plot_val_examples()
            self.log('train/samples/total', len(self.train_dataset))
            self.log('val/samples/total', len(self.val_dataset))

    def setup(self, stage=None):
        self.train_dataset = FaceDataset('data/target', transform=self.transform_train)
        self.val_dataset = FaceDataset('data/target_val/', transform=self.transform_val)

    def plot_examples(self, ds):
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        for i,b in enumerate(ds):
            for j,k in enumerate(b):
                plt.subplot(5,3, 3*i+j+1)
                plt.title(k)
                plt.axis('off')
                plt.imshow(UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(b[k]).permute(1,2,0).numpy()[:,:,::-1])
            if i>=4:
                break
        return fig
    
    def plot_train_examples(self):
        fig = self.plot_examples(self.train_dataset)
        plt.suptitle('Training samples')
        if self.logger is not None:
            plt.close(fig)
            self.logger.experiment.add_figure('train/dataset/samples', fig, 0)
        
    def plot_val_examples(self):
        fig = self.plot_examples(self.val_dataset)
        plt.suptitle('Validation samples')
        if self.logger is not None:
            plt.close(fig)
            self.logger.experiment.add_figure('val/dataset/samples', fig, 0)
    
    def training_step(self, batch, batch_idx):
        # print("\ntraining_step")
        return self._step('train', batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        # print("\nvalidation_step")
        with torch.no_grad():
            output = self._step('val', batch, batch_idx)
            return output
    
    def configure_optimizers(self):
        d=dict()
        d["optimizer"] = torch.optim.RAdam(self.parameters(), lr=self.lr)
#         d["optimizer"] = torch.optim.SGD(self.parameters(), lr=self.lr)


        print("len(self.train_dataset):",len(self.train_dataset))
        print("self.batch_size:",self.batch_size)
        total_steps = self.epochs * len(self.train_dataset)//self.batch_size
        print("total_steps:",total_steps) #drop last = True

        pct_start = 0.1
        div_factor = 100
        final_div_factor = 100
        d["lr_scheduler"] = {"scheduler": torch.optim.lr_scheduler.OneCycleLR(d['optimizer'],
                                            max_lr=self.lr,
                                            total_steps=total_steps,
                                            pct_start=pct_start,
                                            div_factor=div_factor,
                                            final_div_factor=final_div_factor),
                            "frequency": 1,
                            'interval': 'step'
                            }
        return d
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12)

    def predict_batch_exp(self, batch):
        x = self.net[0](batch)
        y = [_(x) for _ in self.net[1:]]
        return prediction2label(y[0]), torch.nn.functional.softmax(y[1]), torch.nn.functional.softmax(y[2])



if torch.cuda.device_count()>0:
    print("Torch             :",torch.__version__)
    print("CUDA devices count:",torch.cuda.device_count())
    print("1st dev name      :", torch.cuda.get_device_name(0))
else:
    print("\n\nNO CUDA!!!\n\n")





backbone='xcit_tiny_12_p8_384_dist'
net = torch.load(backbone+'-tuned.pt')

facerec = FaceRecognition(net, train_size=train_size, test_size=test_size,
                                transform_train=transform_train,
                                transform_val=transform_val,
                                lr=0.001,
                                batch_size=batch_size_freezed)


facerec.eval()
facerec=facerec.cuda()

predictions = {"ImageID":[], "target":[]}

with torch.no_grad():
    image_ids = os.listdir("data/missing")



    

    for img_id in tqdm(image_ids):
        print(img_id,'data/target/'+img_id)

        image100 = cv2.imread('data/target/'+img_id)
        images = torch.stack([transform_val(Image.fromarray(get_target_face(_,image100))) for _ in range(100)]).cuda()
        print(images.shape)
        embs=facerec(images).cpu()

        fn='data/missing/'+img_id
        img = cv2.imread(fn)
        img = transform_val(Image.fromarray(img))

        embedding = facerec(torch.unsqueeze(img, 0).cuda()).cpu()

        ab=torch.abs(embs-embedding)
        ab = torch.mean(ab,axis=1).cpu()
        min_mse_face_no = torch.argmin(ab)
        print(img_id,min_mse_face_no)
        predictions['ImageID'].append(img_id.replace(".jpg", ""))
        predictions['target'].append(min_mse_face_no)

    submission = pd.DataFrame(predictions)
    submission.to_csv(os.path.join("assets", "submission.csv"), index=False)

