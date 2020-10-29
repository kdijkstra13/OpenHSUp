#!/bin/python
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

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from PIL import Image

from hsup.hsup_core import train_upscaling, predict_upscaling
from hsup.types import HSImageDataset
from hsup.models import HSUp, HSConvert
from hsup.util import InputFile, OutputFile, InputFolder, OutputFolder, cube_to_mosaic, cube_to_tiles


def parse_args(args):
    parser = argparse.ArgumentParser(description="Training script for HSUp")
    subparsers = parser.add_subparsers(help="Argument for which task to use", dest="task")
    subparsers.required = True

    train_task = subparsers.add_parser("train-task", help="Use the hsup task")
    train_task.add_argument("--training-file", help="Training input filename where each line contains a filename", required=True, action=InputFile)
    train_task.add_argument("--validation-file", help="Validation input filename where each line contains a filename", action=InputFile)
    train_task.add_argument("--crop", help="Random crop", type=float)
    train_task.add_argument("--init-bilinear", dest="init_bilinear", action="store_true", help="Initialize kernel weight with bilinear interpolation")
    parser.set_defaults(init_bilinear=False)
    train_task.set_defaults(subparser="train-task")

    predict_task = subparsers.add_parser("predict-task", help="Use the predict task")
    predict_task.add_argument("--testing-file", help="Testing input filename where each line contains a filename", action=InputFile)
    predict_task.add_argument("--down-sample", dest="down_sample", action="store_true", help="Down sample before up sampling.")
    parser.set_defaults(down_sample=False)
    predict_task.add_argument("--output-tiles", dest="output_tiles", action="store_true", help="Output tiles instead of mosaic.")
    parser.set_defaults(output_tiles=False)
    predict_task.set_defaults(subparser="predict-task")

    parser.add_argument("--load-snapshot", help="File containing a snapshot of the network", type=str, action=InputFile)
    parser.add_argument("--save-snapshot", help="File to save the snapshot to", type=str, action=OutputFile)
    parser.add_argument("--output-folder", help="Generic output folder to save predictions, loss graphs, snapshots, etc.", type=str, action=OutputFolder)

    parser.add_argument("--filter-size", help="Filter size", type=int, default=8)
    parser.add_argument("--num-filters", help="Capacity", type=int, default=128)
    parser.add_argument("--mosaic-size", help="Size of the mosaic", type=int, default=4)

    parser.add_argument("--gpus", help="GPUs to use (comma separated)", default="0")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=10)
    parser.add_argument("--mini-batches", help="Number of iterations per epoch", type=int)
    parser.add_argument("--epochs", help="Number of Epochs", type=int, default=100)
    parser.add_argument("--learn-rate", help="Learning rate of the optimizer", default=0.001, type=float)
    parser.add_argument("--momentum", help="momentum rate of the optimizer", default=0.9, type=float)
    parser.add_argument("--scheduler", help="scheduler", choices=("steplr"))
    parser.add_argument("--step-size", help="step size for the scheduler", default=500, type=int)
    parser.add_argument("--gamma", help="gamma for the scheduler", default=0.6, type=float)
    parser.add_argument("--weight-decay", help="Weight decay of the optimizer", default=0, type=float)
    parser.add_argument("--sub", help="Subtract this amount", default=512, type=float)
    parser.add_argument("--div", help="Divide by this amount", default=1024, type=float)
    return parser.parse_args(args)


def train_task(args):
    # Create convert model
    convert_model = HSConvert(args.mosaic_size).init_weights()

    # Create model
    if args.load_snapshot:
        model = torch.load(open(args.load_snapshot, "rb"))
    else:
        model = HSUp(mosaic_size=args.mosaic_size, num_filters=args.num_filters, filter_size=args.filter_size)
        if args.init_bilinear:
            model.init_weights()  # Init as bilinear interpolation

    # Create loss
    if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
        torch.save(model, args.save_snapshot)
        print("Warning: saved non-trainable model: {}".format(type(model)))
        return

    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)

    # Create datasets
    training_files = [os.path.join(os.path.dirname(os.path.abspath(args.training_file)), fn) for fn in open(args.training_file, "r").readlines()]
    training_set = HSImageDataset(training_files, mosaic_size=args.mosaic_size,
                                  crop=(args.crop, args.crop) if args.crop is not None else None, sub=args.sub,
                                  div=args.div, convert_model=convert_model)

    validation_files = [os.path.join(os.path.dirname(os.path.abspath(args.validation_file)), fn) for fn in open(args.validation_file, "r").readlines()]
    validation_set = HSImageDataset(validation_files, mosaic_size=args.mosaic_size,
                                    crop=(args.crop, args.crop) if args.crop is not None else None, sub=args.sub,
                                    div=args.div, convert_model=convert_model)

    # Train
    model = train_upscaling(training_set=training_set, validation_set=validation_set,
                            model=model, loss=loss, optimizer=optimizer, scheduler=None,
                            epochs=args.epochs, batch_size=args.batch_size, valid_batch_size=1, mini_batches=args.mini_batches,
                            better_model_cb=lambda model: torch.save(model, args.save_snapshot))
    # Save snapshot
    torch.save(model, args.save_snapshot)


def predict_task_hsup(args, model):
    testing_files = [os.path.join(os.path.dirname(os.path.abspath(args.testing_file)), fn) for fn in open(args.testing_file, "r").readlines()]
    testing_set = HSImageDataset(testing_files, sub=args.sub, div=args.div, down_sample=args.down_sample, mosaic_size=model.mosaic_size)
    output, filenames = predict_upscaling(testing_set, model)
    tile_or_mosaic_func = cube_to_tiles if args.output_tiles else cube_to_mosaic
    for out, filename in tqdm(zip(output, filenames), "Postprocess"):
        out_fn = os.path.join(args.output_folder, filename)
        img = np.ascontiguousarray(tile_or_mosaic_func(out, model.mosaic_size).astype(testing_set.image_type))
        img = Image.fromarray(img)
        img.save(out_fn)


def predict_task(args):
    model = torch.load(open(args.load_snapshot, "rb"))
    predict_task_hsup(args, model)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    else:
        args = [str(x) for x in args]
    args = parse_args(args)

    # Initialize script
    print("Using device(s): {}".format(args.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Call appropriate task
    if args.subparser == "train-task":
        train_task(args)
    elif args.subparser == "predict-task":
        predict_task(args)

    print("\nDone.")


if __name__ == '__main__':
    main()
