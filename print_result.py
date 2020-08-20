import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

home = str(Path.home())
myhost = os.uname()[1]
fs = glob.glob(home + '/ray_results/*/*/best.pth')
sorted(fs)

for f in fs:
    data = torch.load(f, map_location=torch.device('cpu'))
    rs = np.array(data['ITRs'])
    frames = data['frame_count']
    exp_name = f.split('/')[-3]
    with open(f[:-8] + 'params.json', 'r') as fp:
        game = json.load(fp)['game']
    print(
        f"{myhost:>15s} {exp_name:>15s} {game:>15s} {rs.mean():8.2f}\t "
        f"{rs.std():8.2f}\t {rs.max():8.2f}\t {rs.min():8.2f}\t {rs.size:8.2f}\t {frames}")
