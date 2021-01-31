import torch
from torch.utils.data import DataLoader
from mymodel import *


dset_trn = SceneInstanceDataset(
    instance_dir='C:/Users/Hanna/Desktop/geom_survey/cars_train/1ba30d64da90ea05283ffcfc40c29975/',
    instance_idx=0,
    specific_observation_idcs=None,
    img_sidelength=64,
    num_images=-1)

data_loader = DataLoader(dset_trn, batch_size=2)

my_model = MySRN()
my_model.train()
my_model.to(device)

dset_val = SceneInstanceDataset(
    instance_dir='C:/Users/Hanna/Desktop/geom_survey/cars_train/1ba30d64da90ea05283ffcfc40c29975/',
    instance_idx=0,
    specific_observation_idcs=None,
    img_sidelength=64,
    num_images=1)
val_data_loader = DataLoader(dset_val, batch_size=2)

ckpt_path = 'C:/Users/Hanna/Desktop/geom_survey/checkpoint/logs/checkpoints/epoch_0801_iter_100125.pth'
# '/content/gdrive/My Drive/RESEARCH/geom_survey/data/checkpoint/'
if ckpt_path is not None:
    print('Loading model from %s', ckpt_path)
    custom_load(my_model,
          path=ckpt_path)

logging_root = 'C:/Users/Hanna/Desktop/geom_survey/checkpoint/logs'
ckpt_dir = os.path.join(logging_root, 'checkpoints')
events_dir = os.path.join(logging_root, 'events')

cond_mkdir(logging_root)
cond_mkdir(ckpt_dir)
cond_mkdir(events_dir)

# Save command-line parameters log directory.
# with open(os.path.join(logging_root, "params.txt"), "w") as out_file:
#     out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

# Save text summary of model into log directory.
with open(os.path.join(logging_root, "model.txt"), "w") as out_file:
    out_file.write(str(my_model))


start_step = 0
steps_til_val = 1000      #Number of iterations until validation set is run
steps_til_ckpt = 10000    #Number of iterations until checkpoint is saved
l1_weight = 200.0
reg_weight = 1e-3

iter = start_step
epoch = iter // len(dset_trn)
step = 0

optim = torch.optim.Adam(my_model.parameters(), lr = 5e-5)

while epoch <= 1:
    for d in data_loader:

        optim.zero_grad()
        image_pred = my_model(d)

        img_loss = my_model.get_image_loss(image_pred, d['rgb'])
        regu_loss = my_model.get_regularization_loss(image_pred, d['rgb'])

        weighted_img_loss = l1_weight * img_loss
        weighted_regu_loss = reg_weight *regu_loss

        total_loss = img_loss + regu_loss

        total_loss.backward()
        optim.step()

        if iter % steps_til_val == 0:
            print('Running validation set...')

            my_model.eval()
            with torch.no_grad():
                dist_losses = []
                for val in val_data_loader:
                  break
                image_pred = my_model(val)
                #print(type(prediction))

                dist_loss = my_model.get_image_loss(image_pred, val['rgb']).cpu().numpy()
                dist_losses.append(dist_loss)

                my_model.train()
        iter += 1
        step += 1

        if iter % steps_til_ckpt == 0:
            custom_save(my_model,
                os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
                discriminator=None,
                optimizer=optim)
    epoch += 1

custom_save(my_model,
      os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter)),
      discriminator=None,
      optimizer=optim)


import matplotlib.pyplot as plt
# for d in data_loader:
#     break
# print(d)
#pred_image = my_model(d)

# d = data_loader
# for _ in range(5):    
#     pred_image = my_model(d.next())

for i, d in enumerate(data_loader):
    while i < 4:
        pred_image = my_model(d)
        plt.subplot(2, 2, i)
        plt.imshow(pred_image[0, :, :].reshape(64, 64, 3).detach().cpu().numpy())

        
        plt.show()

print('end training')