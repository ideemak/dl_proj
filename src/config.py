#Get data
data_url = 'https://drive.google.com/u/0/uc?id=1L-o6r3_GF-pSaaOTvUOPsgiTKMq_dxIl&export=download'
DATA_DIR_A = './people to banana/banana_data'
DATA_DIR_B = './people to banana/people_data'
name_A = 'banana'
name_B = 'people'
model_name = 'Checkpoints_'+name_A+'_'+name_B+'_'+'model'

#img processing
image_size = 64
batch_size = 5
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

#Model hyperparams
lr = 0.0003
beta1 = 0.5 #Adam hyperparam

#Generator loss coefficients
G_loss_k = 1
cycle_loss_k = 1
identity_loss_k = 3

n_epochs = 100