import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

home = str(Path.home())
myhost = os.uname()[1]
fs = glob.glob(home + '/ray_results/*/*/final.pth')
sorted(fs)

for f in fs:
    data = torch.load(f, map_location=torch.device('cpu'))['FTRs']
    data = np.array(data)
    exp_name = f.split('/')[-3]
    with open(f[:-9] + 'params.json', 'r') as fp:
        game = json.load(fp)['game']

    print(
        f"{myhost:>15s} {exp_name:>15s} {game:>15s} {data.mean():8.2f}\t {data.std():8.2f}\t {data.max():8.2f}\t {data.min():8.2f}\t {data.size:8.2f}\t")
