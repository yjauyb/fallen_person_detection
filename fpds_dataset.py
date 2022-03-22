
from torch.utils.data import Dataset
from os import path
from PIL import Image

class FpdsDataset(Dataset):
    def __init__(self, dataset_dir, transform = None, mode = "train"):
        super().__init__()
        if mode not in ["train", "test", "val"]:
            raise ValueError("the mode can only be one of 'train', 'test', 'val'")
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.image_label = self._load_image_label()
        self.image_dir = path.join(self.dataset_dir, f"fpds_{self.mode}")    
    def _load_image_label(self):
        image_lable = []       
        label_path = path.join(self.dataset_dir, f"fpds_{self.mode}_label.txt")
        with open(label_path) as fh:
            lines = fh.read().splitlines()
            for item in lines:
                image_file_name, label = item.split(";")
                image_lable.append((image_file_name, int(label)))
        return image_lable
    
    def __getitem__(self, index: int):
        image_file_name, label = self.image_label[index]
        image_file_path = path.join(self.image_dir, image_file_name)
        image_data = Image.open(image_file_path).convert("RGB")
        if self.transform is not None:
            image_data = self.transform(image_data)
        return image_data, label

    def __len__(self):
        return len(self.image_label)