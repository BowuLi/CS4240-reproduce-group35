import torch
from vainF_ssim import MS_SSIM
from network import Net
from common_classes import load_data, run_test
import glob
import torch.optim as optim
from torch.utils.data import DataLoader
import shutil
import os
import numpy as np

from datetime import datetime
start_time = datetime.now()

print(torch.cuda.is_available())

opt = {'base_lr': 1e-4}  # Initial learning rate
opt['reduce_lr_by'] = 0.1  # Reduce learning rate by 10 times
opt['atWhichReduce'] = [500000]  # Reduce learning rate at these iterations.
opt['batch_size'] = 8
opt['atWhichSave'] = [2, 100002, 150002, 200002, 250002, 300002, 350002, 400002, 450002, 500002, 550000, 600000, 650002, 700002, 750000,
                      800000, 850002, 900002, 950000, 1000000]  # testing will be done at these iterations and corresponding model weights will be saved.
opt['iterations'] = 1000005  # The model will run for these many iterations.


# Average metrics will be saved here. Please note these are only for supervison. We used MATLAB for final PSNR and SSIM evaluation.
metric_average_file = 'metric_average_fuji100Canon70D.txt'
# Intermediate details for the test images, such as estimated amplification will be saved here.
test_amplification_file = 'test_amplification_fuji100Canon70D.txt'


# These are folders
save_images = 'images_fuji100Canon70D'  # Restored images will be saved here.
# Other details such as loss value and learning rate will be saved in this file.
save_csv_files = 'csv_files_fuji100Canon70D'


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# You would probably like to change it to 0 or some other integer depending on GPU avalability.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

shutil.rmtree(metric_average_file, ignore_errors=True)
shutil.rmtree(test_amplification_file, ignore_errors=True)

shutil.rmtree(save_images, ignore_errors=True)
shutil.rmtree(save_csv_files, ignore_errors=True)

os.makedirs(save_images)
os.makedirs(save_csv_files)

test_files = []
for i in range(10):
    test_files += glob.glob('CanonEOS70D/scene-'+str(i+1)+'/IMG_0005.CR2')

print(test_files)

gt_files = []
for i in range(10):
    gt_files += glob.glob('CanonEOS70D/scene-'+str(i+1)+'/IMG_0001.CR2')

print(gt_files)

dataloader_test = DataLoader(load_data(test_files, gt_files, test_amplification_file, 2,
                             gt_amp=True, training=False), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# load model
device = torch.device("cuda")
model = Net()

checkpoint = torch.load('weights_fuji100/weights_1000000', map_location=device)
model.load_state_dict(checkpoint['model'])

# torch.load() #read weights
print(model)
model = model.to(device)
print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))


run_test(model, dataloader_test, 0, save_images,
         save_csv_files, metric_average_file, 'w', training=True)
# torch.save({'model': model.state_dict()}, os.path.join(
#     save_weights, 'weights_{}'.format(0)))
