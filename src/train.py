import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from model.model import *
from config import *
import numpy as np

bce_loss = nn.BCEWithLogitsLoss()
L1Loss = nn.L1Loss()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
else:
    DEVICE = torch.device("cpu")

train_dl_A = img_to_dl(DATA_DIR_A)
train_dl_B = img_to_dl(DATA_DIR_B)

def fit(model, n_epochs, lr):

    torch.cuda.empty_cache()

    model['G_A2B'].train()
    model['G_B2A'].train()
    model['D_A'].train()
    model['D_B'].train()

    losses_G = []
    losses_D_A = []
    losses_D_B = []

    for epoch in range(n_epochs):

        loss_G_per_epoch = []
        loss_D_A_per_epoch = []
        loss_D_B_per_epoch = []

        for i,(data_A, data_B) in enumerate(tqdm(zip(train_dl_A, train_dl_B), 0)):

            A_real = data_A[0].to(DEVICE)
            B_real = data_B[0].to(DEVICE)

            tensor_ones=torch.ones([A_real.shape[0],1,6,6]).to(DEVICE)
            tensor_zeros=torch.zeros([A_real.shape[0],1,6,6]).to(DEVICE)

            #img generation
            A_fake = model['G_B2A'](B_real)
            B_recovered = model['G_A2B'](A_fake)
            B_fake = model['G_A2B'](A_real)
            A_recovered = model['G_B2A'](B_fake)

            #For identity loss
            A_same = model['G_B2A'](A_real)
            B_same = model['G_A2B'](B_real)

            #Распознаем подлинность картинки
            optimizer['D_A'].zero_grad()
            optimizer['D_B'].zero_grad()

            D_A_real = model['D_A'](A_real)
            D_B_real = model['D_B'](B_real)
            D_A_fake = model['D_A'](A_fake.clone().detach() )
            D_B_fake = model['D_B'](B_fake.clone().detach() )

            D_A_loss = (bce_loss(D_A_real, tensor_ones) + bce_loss(D_A_fake, tensor_zeros))
            D_B_loss = (bce_loss(D_B_real, tensor_ones) + bce_loss(D_B_fake, tensor_zeros))
            
            D_A_loss.backward()
            optimizer['D_A'].step()
            D_B_loss.backward()
            optimizer['D_B'].step()

            loss_D_A_per_epoch.append(D_A_loss.item())
            loss_D_B_per_epoch.append(D_B_loss.item())

            #Generator loss

            optimizer['G_A2B'].zero_grad()
            optimizer['G_B2A'].zero_grad()

            G_A2B_loss = bce_loss(model['D_B'](B_fake), tensor_ones)
            G_B2A_loss = bce_loss(model['D_A'](A_fake), tensor_ones)

            total_cycle_loss = L1Loss(B_real, B_recovered) + L1Loss(A_real, A_recovered)
            all_G_A2B_loss = G_A2B_loss*G_loss_k + total_cycle_loss*cycle_loss_k + L1Loss(B_real, B_same)*identity_loss_k
            all_G_B2A_loss = G_B2A_loss*G_loss_k + total_cycle_loss*cycle_loss_k + L1Loss(A_real, A_same)*identity_loss_k     

            G_loss = all_G_A2B_loss + all_G_B2A_loss

            G_loss.backward()
            optimizer['G_A2B'].step()
            optimizer['G_B2A'].step()

            loss_G_per_epoch.append(G_loss.item())




            #wandb.log({"loss": G_loss})
        losses_G.append(np.mean(loss_G_per_epoch))
        losses_D_A.append(np.mean(loss_D_A_per_epoch))
        losses_D_B.append(np.mean(loss_D_B_per_epoch))
        
        print("Epoch [{}/{}], loss_G: {:.4f}, loss_D_A: {:.4f}, loss_D_B: {:.4f}".format(
            epoch+1, n_epochs, losses_G[-1], losses_D_A[-1], losses_D_B[-1]))
        
        plot_images_test(train_dl_A, train_dl_B)
        name_for_checkpoint = f' {epoch+1}'
        checkpoint(model, name_for_checkpoint)

    return losses_G, losses_D_A, losses_D_B


n_epochs = 100
history = fit(model, n_epochs, lr)

losses_G, losses_D_A, losses_D_B = history

plot_losses(losses_G, losses_D_A)