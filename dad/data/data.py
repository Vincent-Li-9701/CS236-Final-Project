import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import pprint

from dad.config import IMAGES_PATH, LABEL_PATH, IMAGE_SIZE

ATTRIBUTE_DICT = {
    'weather': {'clear': 0, 'rainy': 1, 'snowy': 2, 'overcast': 3, 'partly cloudy': 4, 'foggy': 5, 'undefined': 6},
    'scene': {'city street': 7, 'highway': 8, 'residential': 9, 'parking lot': 10, 'tunnel': 11, 'gas stations': 12, 'undefined': 13},
    'timeofday': {'daytime': 14, 'dawn/dusk': 15, 'night': 16, 'undefined': 17}}
KEY_ATTRIBUTE_DICT = {
    'weather': {'clear': 0, 'rainy': 1, 'snowy': 2, 'cloudy': 3},
    'scene': {'city street': 4, 'highway': 5, 'residential': 6},
    'timeofday': {'daytime': 7, 'dawn/dusk': 8, 'night': 9}}


class ShiftAndScale(object):
    def __call__(self, image):
        return image * 2. - 1.


class DrivingImageDataset(Dataset):
    def __init__(self, folder_path, split, label_path, use_key_attribs=True) -> None:
        super().__init__()
        assert split == "train" or split == "val"
        self.split = split
        self.attributes = KEY_ATTRIBUTE_DICT if use_key_attribs else ATTRIBUTE_DICT
        self.image_paths = os.path.join(folder_path, split)
        if use_key_attribs:
            self.json = self.read_key_attrib_json(label_path)
        else:
            self.json = self.read_label_json(label_path)

        self.preprocess = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
                ShiftAndScale()
            ])

    def get_num_attribs(self):
        num = 0
        for key in ["weather", "scene", "timeofday"]:
            for clss, idx in self.attributes[key].items():
                num += 1
        return num

    def read_key_attrib_json(self, label_path):
        key_attrib_file = os.path.join(label_path,
                                  f"bdd100k_labels_images_{self.split}_small_key.json")
        if not os.path.exists(key_attrib_file):
            json_data = self.read_label_json(label_path)
            key_attrib_data = []
            for data in json_data:
                attribs = data["attributes"]
                # weather
                if attribs["weather"] == "partly cloudy" or attribs["weather"] == "overcast":
                    attribs["weather"] = "cloudy"
                if attribs["weather"] not in self.attributes["weather"]:
                    continue
                if attribs["scene"] not in self.attributes["scene"]:
                    continue
                if attribs["timeofday"] not in self.attributes["timeofday"]:
                    continue
                key_attrib_data.append({"name": data["name"],
                                        "attributes": attribs})
            with open(key_attrib_file, 'w') as f:
                json.dump(key_attrib_data, f, indent=4)
        else:
            key_attrib_data = json.load(open(key_attrib_file))
        return key_attrib_data


    def read_label_json(self, label_path):
        small_label_file = os.path.join(label_path,
                                  f"bdd100k_labels_images_{self.split}_small.json")
        if not os.path.exists(small_label_file):
            label_file = os.path.join(label_path,
                                      f"bdd100k_labels_images_{self.split}.json")
            json_data = []
            f = json.load(open(label_file))
            for data in f:
                json_data.append({"name": data["name"],
                                  "attributes": data["attributes"]})
            with open(small_label_file, 'w') as f:
                json.dump(json_data, f, indent=4)
        else:
            json_data = json.load(open(small_label_file))
        return json_data

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

        y = torch.zeros(self.get_num_attribs())

        y[self.attributes['weather'][doc['attributes']['weather']]] = 1
        y[self.attributes['scene'][doc['attributes']['scene']]] = 1
        y[self.attributes['timeofday'][doc['attributes']['timeofday']]] = 1

        return image, y

    def translate_labels(self, y):
        this_weather, this_scene, this_time = None, None, None
        for weather, idx in self.attributes["weather"].items():
            if y[idx] == 1:
                this_weather = weather
                break
        for scene, idx in self.attributes["scene"].items():
            if y[idx] == 1:
                this_scene = scene
                break
        for time, idx in self.attributes["timeofday"].items():
            if y[idx] == 1:
                this_time = time
                break
        return this_weather, this_scene, this_time

    def plot_img(self, idx):
        img, y = self.__getitem__(idx)
        this_weather, this_scene, this_time = self.translate_labels(y)

        print(f"weather: {this_weather}; scene: {this_scene}; time: {this_time}")
        plt.imshow((img.permute(1, 2, 0) + 1) / 2.)
        plt.show()

    def get_attribute_statistics(self):
        attrib_counter = copy.deepcopy(self.attributes)
        attrib_dist = copy.deepcopy(self.attributes)
        for category, cate_map in attrib_counter.items():
            for subtype in cate_map.keys():
                attrib_counter[category][subtype] = 0
                attrib_dist[category][subtype] = 0

        a = 1. / len(self)
        for i in range(len(self)):
            doc = self.json[i]
            attrib_counter['weather'][doc['attributes']['weather']] += 1.
            attrib_counter['scene'][doc['attributes']['scene']] += 1.
            attrib_counter['timeofday'][doc['attributes']['timeofday']] += 1.
            attrib_dist['weather'][doc['attributes']['weather']] += a
            attrib_dist['scene'][doc['attributes']['scene']] += a
            attrib_dist['timeofday'][doc['attributes']['timeofday']] += a

        pprint.pprint(f"Dataset size: {len(self)}")
        pprint.pprint(attrib_counter)
        pprint.pprint(attrib_dist)

    def generate_random_attributes(self, batch_size, tensor_type, cuda):
        num_attributes = self.get_num_attribs()
        zeros = torch.zeros(batch_size, num_attributes).cuda() if cuda else torch.zeros(batch_size, num_attributes)
        attribs = tensor_type(zeros)
        num_weathers = len(self.attributes['weather'])
        num_scenes = len(self.attributes['scene'])
        first_dim = np.arange(0, batch_size)
        attribs[first_dim, np.random.randint(0, num_weathers, size=batch_size)] = 1.
        attribs[first_dim, np.random.randint(num_weathers, num_weathers + num_scenes, size=batch_size)] = 1.
        attribs[first_dim, np.random.randint(num_weathers + num_scenes, num_attributes, size=batch_size)] = 1.
        return attribs

    def generate_attributes_for_plotting(self, n_row, tensor_type, cuda):
        num_weathers = len(self.attributes['weather'])
        num_scenes = len(self.attributes['scene'])
        num_attributes = self.get_num_attribs()
        zeros = torch.zeros(num_attributes * n_row, num_attributes).cuda() if cuda else torch.zeros(num_attributes * n_row, num_attributes)
        attribs = tensor_type(zeros)

        for i in range(num_attributes):
            first_dim = np.arange(n_row * i, n_row * (i+1))
            attribs[first_dim, i] = 1.

            if i >= num_weathers:
                attribs[first_dim, np.random.randint(0, num_weathers, size=n_row)] = 1.
            if i < num_weathers or i >= num_weathers + num_scenes:
                attribs[first_dim, np.random.randint(num_weathers, num_weathers + num_scenes, size=n_row)] = 1.
            if i < num_weathers + num_scenes:
                attribs[first_dim, np.random.randint(num_weathers + num_scenes, num_attributes, size=n_row)] = 1.
        return attribs

    def generate_attribute(self, weather, scene, timeofday, cuda):
        num_attributes = self.get_num_attribs()
        attrib = torch.zeros(1, num_attributes).cuda() if cuda else torch.zeros(1, num_attributes)
        attrib[:, self.attributes['weather'][weather]] = 1.
        attrib[:, self.attributes['scene'][scene]] = 1.
        attrib[:, self.attributes['timeofday'][timeofday]] = 1.
        return attrib


def test_dataset():
    dataset = DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
    print(f"Dataset size: {len(dataset)}")
    print(dataset.attributes)
    img, y = dataset.__getitem__(0)
    print(img.shape)
    print(y.shape)

    # test pixel value range
    print(f"pixel value: max {torch.max(img)} min {torch.min(img)}")
    # print(img)
    print(y)
    dataset.plot_img(0)
    dataset.plot_img(1)
    dataset.plot_img(2)


def test_generate_attributes():
    dataset = DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
    print(dataset.generate_random_attributes(5, torch.FloatTensor, False))
    print()
    print(dataset.generate_attributes_for_plotting(2, torch.FloatTensor, False))


if __name__ == "__main__":
    # test_dataset()
    # test_generate_attributes()
    # print(generate_random_attributes(10, torch.FloatTensor))

    dataset = DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
    dataset.get_attribute_statistics()
