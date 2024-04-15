#! /usr/bin/env python3
## ---------------------------------------------------------------------------
##
## File: convert_data_to_np.py for Newton Raphson
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Tue Mar 19 11:58:54 2024 Zhijin Li
## Last update Sun Apr 14 23:20:56 2024 Zhijin Li
## ---------------------------------------------------------------------------


import os
import argparse
import numpy as np

from torch.utils.data import DataLoader as dl
from flamby.datasets.fed_heart_disease import FedHeartDisease, HeartDiseaseRaw


if __name__ == '__main__':

  parser = argparse.ArgumentParser(
    'save UCI Heart Disease as numpy arrays.')
  parser.add_argument(
    'save_dir',
    type=str,
    help='directory to save converted numpy arrays as .npy files.'
  )
  args = parser.parse_args()

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)

  for site in range(4):

    for flag in ('train', 'test'):

      # To load data a pytorch dataset
      data = FedHeartDisease(
        center=site,
        train=(flag == 'train')
      )

      # Save training dataset
      data_x = []
      data_y = []
      for (x,y) in dl(data, batch_size=1, shuffle=False, num_workers=0):
        data_x.append(x.cpu().numpy().reshape(-1))
        data_y.append(y.cpu().numpy().reshape(-1))

      data_x = np.array(data_x).reshape(-1, 13)
      data_y = np.array(data_y).reshape(-1, 1)

      print('site {} - {} - variables shape: {}'.format(site, flag, data_x.shape))
      print('site {} - {} - outcomes shape: {}'.format(site, flag, data_y.shape))

      save_x_path = '{}/site-{}.{}.x.npy'.format(args.save_dir, site+1, flag)
      print('saving data: {}'.format(save_x_path))
      np.save(save_x_path, data_x)

      save_y_path = '{}/site-{}.{}.y.npy'.format(args.save_dir, site+1, flag)
      print('saving data: {}'.format(save_y_path))
      np.save(save_y_path, data_y)
