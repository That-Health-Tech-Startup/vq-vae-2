"""
Train a PixelCNN on MNIST using a pre-trained VQ-VAE.
"""
import pandas as pd
import os, sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

from vq_vae_2.examples.mnist.model import Generator, make_vq_vae

BATCH_SIZE = 60000
LR = 1e-3
DEVICE = torch.device('cpu')


def main():
    vae = make_vq_vae()
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu'))
    vae.to(DEVICE)
    vae.eval()

    test_images = load_images(train=True)
    
    
    

    for batch_idx, images in enumerate(load_images()):
        labels = images[1]
        images = images[0].to(DEVICE)
        print(labels)
        for img_set in [images, next(test_images)[0].to(DEVICE)]:
            _, _, encoded = vae.encoders[0](img_set)
            print(encoded.shape)
            print(encoded)
            dic = {"latent space representation" : [f"{x}" for x in encoded], "labels" : labels}
            df = pd.DataFrame(data=dic)
            df.to_csv("MNISTLatent.csv")
            break
        
        break

        





def load_images(train=True):
    while True:
        for data, label in create_data_loader(train):
            yield (data, label)


def create_data_loader(train):
    mnist = torchvision.datasets.MNIST('./data', train=train, download=True,
                                       transform=torchvision.transforms.ToTensor())
    return torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == '__main__':
    main()
