from fpds_dataset import FpdsDataset
import torchvision.transforms as transforms
import torch
from os import path
from collections import defaultdict
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fpds_falldown_swin_model import SwinTransformer
from torchinfo import summary
import time
from utils import predicted_metrics, save_data, plot_loss, plot_metrics
# inputs parameters
dataset_dir = "/home/y/fpds"
model_para_path = "/mnt/E/fall_detection/trained/swin_weights/swin_tiny_patch4_window7_224.pth"
train_save_dir = "/mnt/E/fall_detection/trained"
model_name = "swin_t_adjust"
num_loss_step = 50 #number of steps to average loss for plotting
num_epoch_save = 50 #number of epoches to save loss, model and optimizer state dictionary
workers = 5 # number of subprocesses for data loading
batch_size = 32# samples per batch to load
image_size = (224, 224) # pixels of square image (640 * 640), input size for the model
num_classes = 2
num_epochs = 40 # number of training epochs
num_gpu = 1 # number of GPU
init_val_accuracy = 0.01
lr = 1e-4
weight_decay = 0.0

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

"""
#plot images for view
real_batch = next(iter(train_loader))
print(real_batch[1])
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
"""
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


for name, param in net.named_parameters():
    if name.find("headout") == -1: #set other parametes other than in head to requireds_grad = False, "-1" indicate did not find "head" in the name
            param.requires_grad = False
    #print(name)

#print(net)
#summary(net)
net = net.to(device)
# 3. Define a Loss function and optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=6e-08, weight_decay=0.0, amsgrad=False)

# 4.1 loading parameters, continue Train the network
if model_para_path is not None:
    checkpoint = torch.load(model_para_path)
    #after change model, set the strict argument to False in the load_state_dict() function to ignore non-matching keys.
    #to avoid state dict dimmension error, when modify the model, use new submodule name instead old submodule names
    net.load_state_dict(checkpoint['model'], strict=False) # change when dictionary key is different, e.g. "model_state_dict", "model"


# 4.2 Train the network
losses = []
val_train_metrics = defaultdict(list)
for epoch in range(num_epochs):
    net.train()
    t0 = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward 
        outputs = net(inputs)        
        loss = criterion(outputs, labels)
        #Calculate the gradients
        loss.backward()
        # optimize
        optimizer.step()              
        # Output training stats       
        running_loss += loss.item()       
        if epoch == 0 and i == 0:
            print("[%d/%d] [%d/%d]\tLoss: %.4f" % 
                  (epoch, num_epochs, i+1, len(train_loader), running_loss))
        if i%num_loss_step == num_loss_step-1:        
            print("[%d/%d] [%d/%d]\tLoss: %.4f" % 
                  (epoch, num_epochs, i+1, len(train_loader), running_loss/num_loss_step))
            # Save Losses for plotting later
            losses.append((epoch+(i+1)/len(train_loader), running_loss/num_loss_step))
            running_loss = 0.0          
           
                          
    # save checking point for Inference and/or Resuming Training
    if epoch % num_epoch_save == num_epoch_save-1:
        checkpoint = path.join(train_save_dir, f"{model_name}_checkpoint.pth")
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, checkpoint)
    #save model if get better val accuracy    
    val_metrics = predicted_metrics(val_loader,num_classes, net, device)
    train_metrics = predicted_metrics(sub_train_loader, num_classes, net, device)
    for key in val_metrics.keys():
        val_train_metrics[f"val_{key}"].append(val_metrics[key])
        val_train_metrics[f"train_{key}"].append(train_metrics[key])
    val_train_metrics["epoch"].append(epoch)  
    val_accuracy = val_metrics["accuracy"]       
    if val_accuracy >  init_val_accuracy:
        bestpoint = path.join(train_save_dir, f"{model_name}_bestpoint.pth")
        torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, bestpoint)
        init_val_accuracy = val_accuracy
        print("best val accuracy", init_val_accuracy.item())
    
    t1 = time.time()
    print("time per one epoch:", t1-t0)

    if epoch < num_epochs-1:
        print(f"current epoch: {epoch}, current train metrics: {train_metrics}, current val metrics: {val_metrics}")    
    else:
        print(f"final train metrics: {train_metrics}, final val metrics: {val_metrics}")  

# save trained model
trained_net = path.join(train_save_dir, f"{model_name}_trained_net.pth")
torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, trained_net) 

#save losses and metrics
save_data(train_save_dir, f"{model_name}_losses", losses)
epoch_list = val_train_metrics.pop("epoch")
#unkown reason, torch.round() does not have a decimals option on local, better use torch.round() before tolist()
for key in val_train_metrics.keys():
    val_train_metrics[key] = torch.stack(val_train_metrics[key], dim = 0).tolist()
val_train_metrics["epoch"] = epoch_list
save_data(train_save_dir, f"{model_name}_val_train_metrics", val_train_metrics)
#plot losses versus training iterations
plot_loss(losses, train_save_dir, model_name)
plot_metrics(val_train_metrics, train_save_dir, model_name, mode = "accuracy_recall_precision", class_index =1)
test_metrics = predicted_metrics(test_loader, num_classes, net, device)
for key in test_metrics.keys():
    test_metrics[key] = test_metrics[key].tolist()
    val_metrics[key] = val_metrics[key].tolist()
    train_metrics[key] = train_metrics[key].tolist()
save_data(train_save_dir, f"{model_name}_train_val_test_metrics", {f"train: {train_metrics};", f"val: {val_metrics};", f"test: {test_metrics}"})
print(f"final results:\n train: {train_metrics};\n val: {val_metrics};\n test: {test_metrics}")
print('Finished Training')