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

from train import main
import os
import numpy as np
from skimage.measure import compare_ssim
from hsup.util import read_image, planar_to_pixel, mosaic_to_cube, save_image, cube_to_rgb
from typing import Dict, Tuple, List
from tqdm import tqdm


def train():
    args = ["--output-folder", "output",
            "--save-snapshot", "output/hsup_4x4.pth",
            "--learn-rate", 0.001,
            "--scheduler", "steplr",
            "--gamma", 0.6,
            "--step-size", 100,
            "--momentum", 0.1,
            "--epochs", 5000,
            "--filter-size", 8,
            "--mosaic-size", 4,
            "--num-filters", 128,
            "--sub", 0,
            "--div", 2048,
            "--batch-size", 100,
            "--mini-batches", 10,
            "train-task",
            "--crop", 100,
            "--training-file", "4x4/training.txt",
            "--validation-file", "4x4/validation.txt"]
    main(args)


def train_bl():
    args = ["--output-folder", "output",
            "--save-snapshot", "output/hsup_4x4_bl.pth",
            "--epochs", 0,
            "--filter-size", 3,
            "--mosaic-size", 4,
            "--num-filters", 16,
            "--sub", 0,
            "--div", 2048,
            "train-task",
            "--crop", 100,
            "--init-bilinear",
            "--training-file", "4x4/training.txt",
            "--validation-file", "4x4/validation.txt"]
    main(args)


def upscale(postfix = ""):
    args = ["--output-folder", "output/upscaled",
            "--load-snapshot", "output/hsup_4x4{}.pth".format(postfix),
            "--sub", 0,
            "--div", 2048,
            "predict-task",
            "--down-sample",
            #"--output-tiles",
            "--testing-file", "4x4/validation.txt"]
    main(args)


def demosaick(postfix = ""):
    args = ["--output-folder", "output/demosaicked",
            "--load-snapshot", "output/hsup_4x4{}.pth".format(postfix),
            "--sub", 0,
            "--div", 2048,
            "predict-task",
            #"--output-tiles",
            "--testing-file", "4x4/validation.txt"]
    main(args)


def get_ssim(fns: List[str]) -> Tuple[float, Dict[str, float]]:
    avg_ssim = 0
    ssim_dict = {}
    for fn in tqdm(fns):
        org = read_image("4x4/validation/{}".format(fn))
        res = read_image("output/upscaled/{}".format(fn))
        org = mosaic_to_cube(org)
        res = mosaic_to_cube(res)
        org = planar_to_pixel(org)
        res = planar_to_pixel(res)
        ssim = compare_ssim(org, res, data_range=org.max() - org.min())
        avg_ssim += ssim
        ssim_dict[fn] = ssim
    avg_ssim /= len(fns)
    return avg_ssim, ssim_dict


if __name__ == '__main__':
    os.chdir("./data/")
    # Enable this line to train the HSUp model.
    train()

    # Enable this line to train the bilinear interpolation base line.
    train_bl()

    # Upscale using the validation set.
    upscale()

    # Demosaick using the validation set.
    demosaick()

    # Calculate the SSIM index between original and upscaled images
    avg_ssim, dict_ssim = get_ssim([os.path.basename(x.strip()) for x in open("4x4/validation.txt").readlines()])
    print("SSIM: {:.2f}".format(avg_ssim))

    # Convert a single upscaled image to 8 bits RGB for display
    image = read_image("./output/upscaled/CAM_XIMEACamVIS05458.png")
    image = image.astype(np.float)
    image = cube_to_rgb(mosaic_to_cube(image))
    image = ((image / np.max(image)) * 255).astype(np.uint8)
    save_image(image, "./output/example_upscale.png")

    # Convert a single demosaicked image to 8 bits RGB for display
    image = read_image("./output/demosaicked/CAM_XIMEACamVIS07180.png")
    image = image.astype(np.float)
    image = cube_to_rgb(mosaic_to_cube(image))
    image = ((image / np.max(image)) * 255).astype(np.uint8)
    save_image(image, "./output/example_demosaick.png")
