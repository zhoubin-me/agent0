from collections import deque

import numpy as np
import torch
from lz4.block import decompress
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader, Sampler

from agent0.common.utils import LinearSchedule
from agent0.deepq.config import Config



