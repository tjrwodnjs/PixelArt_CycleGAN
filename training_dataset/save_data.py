import numpy as np
import matplotlib.pyplot as plt

image_number = 50

# Save Domain A Dataset as npy
dataset_A = []

for i in range(1, image_number+1):
    img = plt.imread(f'./training_dataset/trainA/{i}.png')
    dataset_A.append(img)

dataset_A = np.array(dataset_A)
np.save(f'Domain_A_{image_number}_256x256', dataset_A)

print(dataset_A.shape)

# Save Domain B Dataset as npy
dataset_B = []

for i in range(1, image_number+1):
    img = plt.imread(f'./training_dataset/trainB/{i}.png')
    dataset_B.append(img)

dataset_B = np.array(dataset_B)
np.save(f'Domain_B_{image_number}_256x256', dataset_B)

print(dataset_B.shape)

