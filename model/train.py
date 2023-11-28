import numpy as np
import matplotlib.pyplot as plt
import sys
import dezero
from dezero import cuda
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.models import Sequential
from dezero.optimizers import Adam

use_gpu = cuda.gpu_enable
print(use_gpu)

#define generators
G_AB = Sequential(
    #encoding blocks
    L.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    #residual blocks
    L.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    #decoding blocks
    L.Deconv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Deconv2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.tanh,
)

G_BA = Sequential(
    #encoding blocks
    L.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    #residual blocks
    L.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    #decoding blocks
    L.Deconv2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Deconv2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.tanh,
)

#define discriminator
D_A = Sequential(
    L.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, pad=0, nobias=True),
    F.sigmoid,
)

D_B = Sequential(
    L.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, pad=1, nobias=True),
    L.BatchNorm(),
    F.leaky_relu,

    L.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, pad=0, nobias=True),
    F.sigmoid,
)

#load Dataset
dataset_A = np.load('./training_dataset/Domain_A_900_256x256.npy')
dataset_B = np.load('./training_dataset/Domain_B_900_256x256.npy')

data_A = []
data_B = []

# for i in range(min(len(dataset_A), len(dataset_B))):
for i in range(300):
    reshaped_A = np.transpose(dataset_A[i], (2,0,1))
    reshaped_B = np.transpose(dataset_B[i], (2,0,1))
    data_A.append((reshaped_A, reshaped_B))

batch_size = 4

dataloader = DataLoader(data_A, batch_size=batch_size, shuffle=True)

if use_gpu:
    G_AB.to_gpu()
    G_BA.to_gpu()
    D_A.to_gpu()
    D_B.to_gpu()
    dataloader.to_gpu()
    xp = cuda.cupy
else:
    xp = np

#define optimizer
opt_G_AB = Adam(alpha=0.0002, beta1=0.5).setup(G_AB)
opt_G_BA = Adam(alpha=0.0002, beta1=0.5).setup(G_BA)
opt_D_A = Adam(alpha=0.0002, beta1=0.5).setup(D_A)
opt_D_B = Adam(alpha=0.0002, beta1=0.5).setup(D_B)

max_epoch = 100

label_real = xp.ones((batch_size, 1, 30, 30)).astype(int)
label_fake = xp.zeros((batch_size, 1, 30, 30)).astype(int)


loss_d_stack = []
loss_g_stack = []

min_loss = 1e6

for epoch in range(max_epoch):

    print('epoch:', epoch)

    avg_loss_d = 0
    avg_loss_g = 0
    cnt = 0

    dataloader.reset()

    for x_A, x_B in dataloader:
        cnt += 1

        if x_A.shape[0] != batch_size:
          continue

        #============ train D ============#
        # train with real image
        pred_D_A = D_A(x_A)
        D_A_loss = F.mean_squared_error(pred_D_A, label_real)

        pred_D_B = D_B(x_B)
        D_B_loss = F.mean_squared_error(pred_D_B, label_real)

        D_total_loss = D_A_loss + D_B_loss
        D_A.cleargrads()
        D_B.cleargrads()
        G_AB.cleargrads()
        G_BA.cleargrads()
        D_total_loss.backward()
        opt_D_A.update()
        opt_D_B.update()

        # train with fake image
        fake_x_A = G_BA(x_B)
        pred_D_A = D_A(fake_x_A)
        D_A_loss = F.mean_squared_error(pred_D_A, label_fake)

        fake_x_B = G_AB(x_A)
        pred_D_B = D_B(fake_x_B)
        D_B_loss = F.mean_squared_error(pred_D_B, label_fake)

        D_total_loss = D_A_loss + D_B_loss
        D_A.cleargrads()
        D_B.cleargrads()
        G_AB.cleargrads()
        G_BA.cleargrads()
        D_total_loss.backward()
        opt_D_A.update()
        opt_D_B.update()

        #============ train G ============#
        # train A-B-A cycle
        fake_B = G_AB(x_A)
        pred_fake_D_B = D_B(fake_B)
        reconst_A = G_BA(fake_B)

        G_loss = F.mean_squared_error(pred_fake_D_B, label_real) + F.mean_squared_error(x_A, reconst_A)

        D_A.cleargrads()
        D_B.cleargrads()
        G_AB.cleargrads()
        G_BA.cleargrads()
        G_loss.backward()
        opt_G_AB.update()
        opt_G_BA.update()

        # train B-A-B cycle
        fake_A = G_BA(x_B)
        pred_fake_D_A = D_A(fake_A)
        reconst_B = G_AB(fake_A)

        G_loss = F.mean_squared_error(pred_fake_D_A, label_real) + F.mean_squared_error(x_B, reconst_B)

        D_A.cleargrads()
        D_B.cleargrads()
        G_AB.cleargrads()
        G_BA.cleargrads()
        G_loss.backward()
        opt_G_AB.update()
        opt_G_BA.update()

        avg_loss_d += D_total_loss.data
        avg_loss_g += G_loss.data
        
        interval = 10 if use_gpu else 5
        if cnt % interval == 0:
            print('loss_d: {:.4f}, loss_g: {:.4f}'.format(avg_loss_d / cnt, avg_loss_g / cnt))
            loss_d_stack.append(avg_loss_d / cnt)
            loss_g_stack.append(avg_loss_g / cnt)

        if G_loss.data < min_loss:
            min_loss = G_loss.data
            G_AB.save_weights(f'G_AB_manual_epoch:{epoch}.npz')
            D_A.save_weights(f'D_A_manual_epoch:{epoch}.npz')
            D_B.save_weights(f'D_B_manual_epoch:{epoch}.npz')

            G_AB.to_gpu()
            D_A.to_gpu()
            D_B.to_gpu()
