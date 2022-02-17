import matplotlib.pyplot as plt
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import numpy as np
import torch
from config import *
from model.discriminator import D_A, D_B
from model.generator import G_A2B, G_B2A


def device():
    '''cuda or cpu'''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


model = {
    'G_A2B': G_A2B.to(device()),
    'G_B2A': G_B2A.to(device()),
    'D_A': D_A.to(device()),
    'D_B': D_B.to(device())
}

optimizer = {
    'G_A2B': torch.optim.Adam(model['G_A2B'].parameters(), lr=lr, betas=(beta1, 0.999)),
    'G_B2A': torch.optim.Adam(model['G_B2A'].parameters(), lr=lr, betas=(beta1, 0.999)),
    'D_A': torch.optim.Adam(model['D_A'].parameters(), lr=lr, betas=(beta1, 0.999)),
    'D_B': torch.optim.Adam(model['D_B'].parameters(), lr=lr, betas=(beta1, 0.999))
}


def plot_losses(losses_D, losses_G):
    plt.figure(figsize=(15, 6))
    plt.plot(losses_D, '-')
    plt.plot(losses_G, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses')
    plt.show()


def img_to_dl(path, image_size, batch_size=1):
    stats=(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    ds = ImageFolder(path, transform=tt.Compose([
    tt.Resize(image_size),
    tt.CenterCrop(image_size),
    tt.ToTensor(),
    tt.Normalize(*stats)]))

    dl = DataLoader(ds, batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return dl


#Checkpoint
def checkpoint(model, name):
    G_A2B, G_B2A, D_A, D_B = model['G_A2B'], model['G_B2A'], model['D_A'], model['D_A']
    torch.save(G_A2B.state_dict(), "model_checkpoint/"+model_name+'/'+'epoch'+name+'/'+'epoch'+name+"_G_A2B.pt")
    torch.save(G_B2A.state_dict(), "model_checkpoint/"+model_name+'/'+'epoch'+name+'/'+'epoch'+name+"_G_B2A.pt")
    torch.save(D_A.state_dict(), "model_checkpoint/"+model_name+'/'+'epoch'+name+'/'+'epoch'+name+"_D_A.pt")
    torch.save(D_B.state_dict(), "model_checkpoint/"+model_name+'/'+'epoch'+name+'/'+'epoch'+name+"_D_B.pt")
    

#Load checkpoint
def load_checkpoint(model, name):
    DEVICE = device()
    model['G_A2B'].load_state_dict(torch.load("model_checkpoint/"+model_name+'/'+name+'/'+name+"_G_A2B.pt", map_location=DEVICE))
    model['G_B2A'].load_state_dict(torch.load("model_checkpoint/"+model_name+'/'+name+'/'+name+"_G_B2A.pt", map_location=DEVICE))
    model['D_A'].load_state_dict(torch.load("model_checkpoint/"+model_name+'/'+name+'/'+name+"_D_A.pt", map_location=DEVICE))
    model['D_B'].load_state_dict(torch.load("model_checkpoint/"+model_name+'/'+name+'/'+name+"_D_B.pt", map_location=DEVICE))





#код plot_images_test взят с https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def plot_images_test(dataloader_test_A, dataloader_test_B):
    DEVICE = device()
    batch_a_test = next(iter(dataloader_test_A))[0].to(DEVICE)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = model['G_A2B'](batch_a_test).cpu().detach()


    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(vutils.make_grid((real_a_test[:batch_size]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real "+name_A)
    plt.show()

          
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(vutils.make_grid((fake_b_test[:batch_size]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake "+name_B)
    plt.show()


    batch_b_test = next(iter(dataloader_test_B))[0].to(DEVICE)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = model['G_B2A'](batch_b_test).cpu().detach()
    
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(vutils.make_grid((real_b_test[:batch_size]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Real "+name_B)
    plt.show()


    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(vutils.make_grid((fake_a_test[:batch_size]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    plt.axis("off")
    plt.title("Fake "+name_A)
    plt.show()


