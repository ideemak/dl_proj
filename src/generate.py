import torch
from torchvision.utils import save_image
from utils import img_to_dl, load_checkpoint
from model import model


def generate(model, dl, domain):
    if domain == 'A':
        A2B = model['G_A2B']
        with torch.no_grad():
            for n, i in enumerate(dl):
                A2B.eval()
                out = A2B(i[0])
                save_image(out, f'/generated_img/img_A2B_{n+1}.png', normalize=True)
    else:
        B2A = model['G_B2A']
        with torch.no_grad():
            for n, i in enumerate(dl):
                B2A.eval()
                out = B2A(i[0])
                save_image(out, f'/generated_img/img_B2A_{n+1}.png', normalize=True)

path = '/input_img/'
image_size = 64

load_checkpoint('epoch 50')
dl = img_to_dl(path, image_size)
generate(model, dl, 'A')