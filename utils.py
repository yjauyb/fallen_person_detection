import glob
from os import path
import pandas as pd
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt

def convert_fpds_dataset(data_dir, new_dir, image_label_name):
    image_lable = []
    for label_path in glob.iglob(data_dir + '/**/*.txt', recursive= True):
        image_path = label_path.split(".")[0] + ".png"
        image_file_name = label_path.split(path.sep)[-1].split(".")[0] + ".png"
        mark = 0 # when mark == 0, not fall down; mark ==1 fall down
        with open(label_path) as fh:
            lines = fh.read().splitlines()            
            for item in lines:                
                label = int(item.split()[0])
                if label == 1:
                    mark = 1
                    break
        image_lable.append((image_file_name, mark))

    image_label_df = pd.DataFrame(image_lable)
    image_lable_path =  path.join(new_dir, image_label_name+".txt")
    image_label_df.to_csv(image_lable_path, sep = ";", index = False, header= False)

#data_dir = "/mnt/E/fall_detection/data/fpds/valid"
#new_dir = "/mnt/E/fall_detection/data/fpds"
#image_label_name = "fpds_valid_label"   
#convert_fpds_dataset(data_dir, new_dir, image_label_name)

def copy_subdir_images_together(data_dir, desti_dir):
    for image_path in glob.iglob(data_dir + '/**/*.png', recursive= True):
        new_path = shutil.copy(image_path, desti_dir)

#data_dir = "/mnt/E/fall_detection/data/fpds/valid"
#desti_dir = "/home/y/fpds/fpds_val"
#copy_subdir_images_together(data_dir, desti_dir)


def predicted_metrics(data_loader, num_classes, net, device):
    
    net = net.eval()    
    with torch.no_grad():
        correct = 0    
        total = 0
        total_sample_per_class = torch.zeros(num_classes).to(device) #total number of samples in each class, true positives + false negatives
        total_true_positive_per_class = torch.zeros(num_classes).to(device) #total number of samples in each class was corrected predicted as in this class, true positive
        total_predicted_per_class = torch.zeros(num_classes).to(device) # total number of samples was predicted in each classes, true positives + false positives
        for i, data in enumerate(data_loader, 0):
            # get the images; data is a list of [images, labels]            
            images, labels = data[0].to(device), data[1].to(device) 
            # forward            
            outputs = net(images)
            #get index of the first maximal value of predicted results for each sample, that is, predicted label.
            predicted_label = torch.argmax(outputs, dim = 1)
            total += labels.shape[0]
            correct += (predicted_label == labels).sum()
            accuracy = correct/total

            predicted  = torch.nn.functional.one_hot(predicted_label, num_classes) #shape (B, C)
            labels = torch.nn.functional.one_hot(labels, num_classes) #shape (B, C)
            num_sample_per_class = torch.sum(labels, dim=0) #shape(C), number of samples in each class in the batch, that is: true positives + false negatives
            true_positive_per_class = torch.sum(predicted*labels, dim= 0) #shape(C), in the batch, the sample in each class was corrected predicted as in this class, that is: true positive
            num_pridicted_per_class = torch.sum(predicted, dim=0) #shape(C), in the batch, the number of sample was predicted was in each classes, that is: true positives + false positives
            total_sample_per_class += num_sample_per_class
            total_true_positive_per_class += true_positive_per_class
            total_predicted_per_class += num_pridicted_per_class

        recall = total_true_positive_per_class/total_sample_per_class
        precision = total_true_positive_per_class/total_predicted_per_class
        f1_score = (precision * recall *2) / (precision + recall) 
        return {"accuracy" : accuracy, "recall": recall, "precision" : precision, "f1_score" : f1_score}

def save_data(directory, file_name, data):
    data_df = pd.DataFrame(data)
    data_path =  path.join(directory, file_name+".txt")
    data_df.to_csv(data_path, sep = ";", index = False, header= True)

def plot_loss(losses, train_save_dir, model_name):
    #losses, [(epoch, loss), ......]
    epoch, loss = list(zip(*losses))
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    plt.plot(epoch, loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.show()
    plt.savefig(path.join(train_save_dir, f"{model_name}_losses.png"))
def plot_metrics(val_train_metrics, train_save_dir, model_name, mode = "accuracy_recall_precision", class_index =1):
    #val_train_metrics, defaultdict[list], val_accuracy, val_recall, val_precision, val_f1_score, train_accuracy, ...., epoch
    epoch = val_train_metrics["epoch"]
    val_accuracy = np.squeeze(np.array(val_train_metrics["val_accuracy"]))       
    train_accuracy = np.squeeze(np.array(val_train_metrics["train_accuracy"]))
    val_recall = np.array(val_train_metrics["val_recall"])[:, class_index]
    train_recall = np.array(val_train_metrics["train_recall"])[:, class_index]
    val_precsion = np.array(val_train_metrics["val_precision"])[:, class_index]
    train_precsion = np.array(val_train_metrics["train_precision"])[:, class_index]
    plt.figure(figsize=(10,5))
    plt.title("metrics During Training")
    plt.plot(epoch, val_accuracy, label="val_accuracy")
    plt.plot(epoch, train_accuracy, label= "train_accuracy")
    plt.plot(epoch, val_recall, label= "val_recall")
    plt.plot(epoch, train_recall, label= "train_recall")
    plt.plot(epoch, val_precsion, label= "val_precsion")
    plt.plot(epoch, train_precsion, label= "train_precsion")
    plt.xlabel("Epoch")
    plt.ylabel("metrics")
    plt.legend()
    #plt.show()
    plt.savefig(path.join(train_save_dir, f"{model_name}_val_train_accuracy.png"))
