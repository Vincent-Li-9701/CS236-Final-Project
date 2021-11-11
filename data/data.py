import torch 
import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

NUM_ATTRIBUTE = 18

class DrivingImageDataset(Dataset):
    def __init__(self, folder_path, split, label_path) -> None:
        super().__init__()
        self.image_paths = os.path.join(folder_path, split)
        self.json = json.load(open(label_path))
        self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        
        self.attributes = self.collect_all_attributes()
    
    def collect_all_attributes(self):
        attributes = {}
        count = 0
        attributes['weather'] = {}
        attributes['scene'] = {}
        attributes['timeofday'] = {}

        for doc in self.json:
            if doc['attributes']["weather"] not in attributes['weather']:
                attributes['weather'][doc['attributes']['weather']] = count
                count += 1

            if doc['attributes']['scene'] not in attributes['scene']:
                attributes['scene'][doc['attributes']['scene']] = count
                count += 1

            if doc['attributes']['timeofday'] not in attributes['timeofday']:
                attributes['timeofday'][doc['attributes']['timeofday']] = count
                count += 1
        
        return attributes


    def __len__(self):
        return len(self.json)


    def __getitem__(self, idx):
        file_name = os.path.join(self.image_paths, self.json[idx]["name"])
        image = Image.open(file_name)
        image = self.preprocess(image)
        doc = self.json[idx]

        y = torch.zeros(NUM_ATTRIBUTE)

        y[self.attributes['weather'][doc['attributes']['weather']]] = 1
        y[self.attributes['scene'][doc['attributes']['scene']]] = 1
        y[self.attributes['timeofday'][doc['attributes']['timeofday']]] = 1
        
        return image, y


def main():
    dataset = DrivingImageDataset(folder_path="../bdd100k/images/100k/", split='train', label_path='../bdd100k/labels/bdd100k_labels_images_train.json')
    img, y = dataset.__getitem__(0)
    print(dataset.attributes)
    print(img.shape)
    print(y.shape)


if __name__ == "__main__":
    main()
    




