import torch
from discriminator import D_A, D_B
from generator import G_A2B, G_B2A
from config import lr, beta1

if torch.cuda.is_available():
    DEVICE = torch.device("cuda") 
else:
    DEVICE = torch.device("cpu")


model = {
    'G_A2B': G_A2B.to(DEVICE),
    'G_B2A': G_B2A.to(DEVICE),
    'D_A': D_A.to(DEVICE),
    'D_B': D_B.to(DEVICE)
}

optimizer = {
    'G_A2B': torch.optim.Adam(G_A2B.parameters(), lr=lr, betas=(beta1, 0.999)),
    'G_B2A': torch.optim.Adam(G_B2A.parameters(), lr=lr, betas=(beta1, 0.999)),
    'D_A': torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999)),
    'D_B': torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))
}
