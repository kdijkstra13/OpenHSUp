# Copyright (C) 2020 Klaas Dijkstra
#
# This file is part of OpenHSUp.
#
# OpenHSUp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# OpenHSUp is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with OpenHSUp.  If not, see <https://www.gnu.org/licenses/>.

from torch.utils.data import Dataset
from typing import Tuple
import os
import random
import numpy as np
from typing import List, Optional
from PIL import Image
from hsup.models import HSConvert
from hsup.util import down_sample_mosaic
import torch


class HSImageDataset(Dataset):

    def __init__(self, image_list: List[str], mosaic_size: int=1, crop: Optional[Tuple[int, int]]=None, sub=1024, div=2048, repeat=1, image_type=np.int16, convert_model: HSConvert = None, device="cuda:0", down_sample: bool = False):
        """
        A dataset initialized with filenames. Each filename should point to a raw mosaic image.

        :param image_list: List of masaic data
        :param mosaic_size: Size of the mosaic (e.g. 4)
        :param crop: Size of the random crop
        :param sub: Subtract this value from each pixel
        :param div: Divide each pixel by this value
        :param repeat: Repeat a single image multiple times (use this when the mini-batch size is larger then the amount of data)
        :param image_type: Type of the image
        :param convert_model: A deep learning model for converting from mosaic to cube representation
        :param device: The cuda device
        :param down_sample: Determines is the input is downsampled before training.
            When true this represents Upscaling from the original paper, otherwise Demosaicking.
        """
        self.images = [fn.strip() for fn in image_list]
        self.crop = crop
        self.sub = sub
        self.div = div
        self.repeat = repeat
        self.mosaic_size = mosaic_size
        self.image_type = image_type
        self.device = device
        self.convert_model = convert_model
        self.down_sample = down_sample
        self.convert_model.to(self.device) if self.convert_model is not None else None

    def create_input_target_pair(self, input):
        input = np.expand_dims(input, axis=0) # could be done on a minibatch
        input_tensor = torch.Tensor(np.stack([down_sample_mosaic(x, self.mosaic_size) for x in input])).to(self.device)
        target_tensor = torch.Tensor(input).to(self.device)
        target_tensor = self.convert_model(target_tensor)
        return input_tensor[0], target_tensor[0]

    def __getitem__(self, index):
        index = index // self.repeat
        try:
            img = Image.open(self.images[index])
            img = np.array(img, dtype=self.image_type)
        except TypeError:
            print("File {} if corrupt.".format(self.images[index]))
            raise

        img = np.expand_dims(img, axis=2) if len(img.shape) == 2 else img

        if self.crop is not None:
            h, w, _ = img.shape
            tl = [random.randint(mn, mx) for mn, mx in zip([0, 0], [h//self.mosaic_size - self.crop[0], w//self.mosaic_size - self.crop[1]])]
            tl = [int(x * self.mosaic_size) for x in tl]
            br = [int(x + c * self.mosaic_size) for x, c in zip(tl, self.crop)]
            img = img[tl[0]:br[0], tl[1]:br[1], :]

        # Convert to planar format
        img = ((np.transpose(img, (2, 0, 1)) - self.sub) / self.div).astype(np.float32)

        # Down sample if needed
        if self.down_sample:
            img = down_sample_mosaic(img, self.mosaic_size)

        # Return input, target pair
        if self.convert_model is not None:
            return self.create_input_target_pair(img)
        else:
            return torch.Tensor(img).to(self.device), os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images) * self.repeat
