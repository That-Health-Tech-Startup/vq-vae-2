import torch
import torchvision.datasets
import matplotlib as plt
from autoencoder import AutoEncoder

def main():
    # load mnist data transformed into tensor
    data = torchvision.datasets.MNIST(root = "mnist", download = True, transform = torchvision.transforms.ToTensor())
    mnist_dataloaders = torch.utils.data.DataLoader(data, batch_size = 64)
    
    # pull one batch for testing autoencoder
    test_batch = next(iter(mnist_dataloaders))
    
    # instantiate autoencoder model and adam optimizer
    model = AutoEncoder(1, 5)
    param = model.parameters()
    optimizer = torch.optim.Adam(param)
    loss_func = torch.nn.MSELoss(reduction = 'sum')
    losses = []
    
    # 250 epochs
    for i in range(300):
        recon = model(test_batch[0])
        loss = loss_func(recon, test_batch[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    
    rand_batch = next(iter(mnist_dataloaders))
    plt.pyplot.imshow(model(rand_batch[0][1].unsqueeze(0)).detach().numpy().squeeze())
    plt.pyplot.imshow(rand_batch[0][1].detach().numpy().squeeze())    
    
    
if __name__ == "__main__":
    main()