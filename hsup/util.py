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

import os
import argparse
import numpy as np
from PIL import Image
from hsup.models import HSConvert
import torch


class InputFile(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        values = os.path.normpath(values)
        if not os.path.isfile(values):
            raise Exception("File does not exist: {}".format(os.path.abspath(values)))
        setattr(args, self.dest, values)


class InputFolder(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        values = os.path.normpath(values)
        if not os.path.isdir(values):
            raise Exception("File does not exist: {}".format(os.path.abspath(values)))
        setattr(args, self.dest, values)


class OutputFile(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        values = os.path.normpath(values)
        if os.path.dirname(values) != "":
            os.makedirs(os.path.dirname(values), exist_ok=True)
        file = open(values, "w+")
        if not file:
            raise Exception("Cannot create file: {}".format(os.path.abspath(values)))
        setattr(args, self.dest, values)


class OutputFolder(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        values = os.path.normpath(values)
        os.makedirs(values, exist_ok=True)
        setattr(args, self.dest, values)


def down_sample_mosaic(input: np.ndarray, mosaic_size: int) -> np.ndarray:
    c, h, w = input.shape
    s = mosaic_size
    assert c == 1, "Only single channel images supported"
    output = np.empty([1, h // mosaic_size, w // mosaic_size])
    for y in range(h // mosaic_size):
        for x in range(w // mosaic_size):
            output[:, y, x] = input[:, y * s + y % s, x * s + x % s]
    return output


def cube_to_mosaic(input: np.ndarray, mosaic_size: int) -> np.ndarray:
    output = np.empty((input.shape[1] * mosaic_size, input.shape[2] * mosaic_size), dtype=input.dtype)
    for c in range(input.shape[0]):
        for y in range(input.shape[1]):
            for x in range(input.shape[2]):
                output[(y * mosaic_size) + c // mosaic_size, (x * mosaic_size) + c % mosaic_size] = input[c, y, x]
    return output


@torch.no_grad()
def mosaic_to_cube(img : np.ndarray) -> np.ndarray:
    dtype = img.dtype
    convert = HSConvert(mosaic_size=4)
    img_tensor = torch.from_numpy(np.expand_dims(np.expand_dims(img, axis=0), axis=0)).float()
    img_tensor = convert(img_tensor)
    return img_tensor.numpy().astype(dtype)[0]


def cube_to_tiles(input: np.ndarray, mosaic_size: int) -> np.ndarray:
    output = np.empty((input.shape[1] * mosaic_size, input.shape[2] * mosaic_size), dtype=input.dtype)
    _, h, w = input.shape
    for y in range(mosaic_size):
        for x in range(mosaic_size):
            xx = x * w
            yy = y * h
            c = y * mosaic_size + x
            output[yy:yy+h, xx:xx+w] = input[c]
    return output


def cube_to_rgb(image: np.ndarray) -> np.ndarray:
    nc = image.shape[0]
    img = np.empty((3, image.shape[1], image.shape[2]), dtype=image.dtype)
    img[2] = np.average(image[0:nc // 3], axis=0)
    img[1] = np.average(image[nc // 3:2 * (nc // 3)], axis=0)
    img[0] = np.average(image[2 * (nc // 3)], axis=0)
    img = np.moveaxis(img, 0, 2)
    return img


def read_image(fn: str) -> np.ndarray:
    img = Image.open(fn)
    img = np.array(img)
    return img


def save_image(img: np.ndarray, fn: str):
    img = Image.fromarray(img)
    img.save(fn)
    print(f"Saved file: {os.path.abspath(fn)}")


def pixel_to_planar(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (2, 0, 1))


def planar_to_pixel(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (1, 2, 0))
