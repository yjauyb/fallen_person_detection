from fpds_dataset import FpdsDataset
import torchvision.transforms as transforms
import torch
from os import path
import numpy as np
from PIL import Image
from fpds_falldown_swin_model import SwinTransformer
from utils import predicted_metrics, save_data
# inputs parameters
dataset_dir = "/home/y/fpds"
model_para_path = "/mnt/E/fall_detection/trained/train_200_epoch/swin_t_adjust_trained_net.pth"
train_save_dir = "/mnt/E/fall_detection/trained"
model_name = "swin_t_adjust"
workers = 5 # number of subprocesses for data loading
batch_size = 32# samples per batch to load
image_size = (224, 224) # pixels of square image (640 * 640), input size for the model
num_classes = 2
num_gpu = 1 # number of GPU


# 1. data preprocess and load

train = FpdsDataset(dataset_dir, transform = transforms.Compose([
            transforms.Resize(int((256 / 224) * image_size[0]), interpolation=Image.BICUBIC), # when image size is 224,  to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(image_size),  #according to swin, when image_size is 384, not crop step, only same size resize  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]), mode = "train")
val = FpdsDataset(dataset_dir, transform = transforms.Compose([
            transforms.Resize(int((256 / 224) * image_size[0]), interpolation=Image.BICUBIC), # when image size is 224,  to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(image_size),  #according to swin, when image_size is 384, not crop step, only same size resize  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]), mode = "val")

test = FpdsDataset(dataset_dir, transform = transforms.Compose([
            transforms.Resize(int((256 / 224) * image_size[0]), interpolation=Image.BICUBIC), # when image size is 224,  to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(image_size),  #according to swin, when image_size is 384, not crop step, only same size resize  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]), mode = "test")

sub_train = torch.utils.data.Subset(train, np.random.choice(len(train), len(val), replace = False)) 

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, 
                                         shuffle = True, num_workers = workers, pin_memory=True, drop_last= True)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, 
                                         shuffle = False, num_workers = workers, pin_memory=False, drop_last= False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, 
                                         shuffle = False, num_workers = workers, pin_memory=False, drop_last= False)
sub_train_loader = torch.utils.data.DataLoader(sub_train, batch_size = batch_size, 
                                         shuffle = False, num_workers = workers, pin_memory=False, drop_last= False)

device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

#2. loading model

#swin_tiny_patch4_window7_224
net = SwinTransformer(img_size=image_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=num_classes,
                                embed_dim=96,
                                depths=[2, 2, 6, 2],
                                num_heads=[3, 6, 12, 24],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

net = net.to(device)
#loading parameters
checkpoint = torch.load(model_para_path)
net.load_state_dict(checkpoint["model_state_dict"], strict=True)

val_metrics = predicted_metrics(val_loader,num_classes, net, device)
train_metrics = predicted_metrics(sub_train_loader, num_classes, net, device)
test_metrics = predicted_metrics(test_loader, num_classes, net, device)
#unkown reason, torch.round() does not have a decimals option on local
for key in test_metrics.keys(): 
    test_metrics[key] = test_metrics[key].tolist()
    val_metrics[key] = val_metrics[key].tolist()
    train_metrics[key] = train_metrics[key].tolist()
save_data(train_save_dir, f"{model_name}_train_val_test_metrics", {f"train: {train_metrics};", f"val: {val_metrics};", f"test: {test_metrics}"})
print(f"final results:\n train: {train_metrics};\n val: {val_metrics};\n test: {test_metrics}")
print('Finished Training')