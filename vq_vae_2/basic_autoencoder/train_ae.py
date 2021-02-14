import torch.cuda
import torch.optim
import torchvision.datasets
import torch.utils.data
from autoencoder import AutoEncoder

def main():
    # load mnist data transformed into tensor
    data = torchvision.datasets.MNIST(root = "mnist", download = True, transform = torchvision.transforms.ToTensor())
    mnist_dataloaders = torch.utils.data.DataLoader(data, batch_size = 64)
    
    # pull one batch for testing autoencoder
    test_batch = next(iter(mnist_dataloaders))
    
    model = AutoEncoder(1, 5)
    param = model.parameters()
    optimizer = torch.optim.Adam(param)
    

    
if __name__ == "__main__":
    main()