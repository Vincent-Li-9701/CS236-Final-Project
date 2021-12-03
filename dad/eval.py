from dad.data import *
from dad.config import *

import os
import torch
from PIL import Image
import numpy as np
import json


def prepare_real_datasets(path, num_data=2048, offset=0):
    os.makedirs(path, exist_ok=True)
    image_path = os.path.join(path, "images")
    os.makedirs(image_path, exist_ok=True)
    dataset = data.DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
    json_data = dataset.json[offset:offset+num_data]

    for i in range(num_data):
        print(i, end="r")
        x, y = dataset[offset + i]
        x = (x.numpy() + 1.) / 2. * 255.
        x = x.astype(np.uint8)
        img = Image.fromarray(x.transpose(1, 2, 0))
        img.save(os.path.join(image_path, f"{i}.png"))
    print()

    with open(os.path.join(path, "eval_data.json"), 'w') as f:
        json.dump(json_data, f, indent=4)


def prepare_model_datasets(generator, dataset, path, real_json_path, num_data=2048, batch_size=64, cuda=True):
    os.makedirs(path, exist_ok=True)
    image_path = os.path.join(path, "images")
    os.makedirs(image_path, exist_ok=True)
    num_attributes = dataset.get_num_attribs()
    json_data = json.load(open(real_json_path))
    print(len(json_data))
    device = "cuda" if cuda else "cpu"

    for i in range(0, num_data, batch_size):
        print(i)
        label = torch.zeros(batch_size, num_attributes).cuda() if cuda else torch.zeros(batch_size, num_attributes)
        for j in range(batch_size):
            attribs = json_data[i + j]["attributes"]
            weather, scene, timeofday = attribs["weather"], attribs["scene"], attribs["timeofday"]
            label[j, dataset.attributes['weather'][weather]] = 1.
            label[j, dataset.attributes['scene'][scene]] = 1.
            label[j, dataset.attributes['timeofday'][timeofday]] = 1.

        z = torch.randn(batch_size, 100, device=device)
        gen_imgs = generator(z, label)
        for j in range(batch_size):
            x = gen_imgs[j, :].detach().cpu().numpy()
            x = (x + 1.) / 2. * 255.
            x = x.astype(np.uint8)
            img = Image.fromarray(x.transpose(1, 2, 0))
            img.save(os.path.join(image_path, f"{i + j}.png"))


def generate_Dc_Gdc(cuda=True):
    from dad.model.Dc_Gdc import cDCGenerator, Discriminator
    dataset = data.DrivingImageDataset(folder_path=IMAGES_PATH, split='train', label_path=LABEL_PATH, use_key_attribs=True)
    num_attributes = dataset.get_num_attribs()

    epoch = 29
    model_path = os.path.join(LOG_PATH, "Dc_Gdc", "ckpt", f"generator_{epoch}.pth")

    generator = cDCGenerator(num_attributes, 100)
    generator.load_state_dict(torch.load(model_path))
    if cuda:
        generator.cuda()
    generator.eval()
    prepare_model_datasets(generator, dataset, "/home/anthony/eval/Dc_Gdc/", "/home/anthony/eval/real/eval_data.json", cuda=cuda)



# prepare_real_datasets("/home/anthony/eval/real2/", 2048, offset=2048)
generate_Dc_Gdc()
