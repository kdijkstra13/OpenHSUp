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

import torch.nn
from hsup.models import HSUp
from hsup.types import HSImageDataset
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm import tqdm


def validate(set_loader: DataLoader, model: HSUp, loss: nn.Module):
    result, idx = 0, 0
    with torch.no_grad():
        for input, target in set_loader:
            output = model(input)
            ls = loss(output, target)
            result += ls.item()
            idx += 1
        return result / idx


def predict_upscaling(testing_set: HSImageDataset, model: HSUp, device: str = "cuda:0"):
    model.to(device)
    testing_set_loader = DataLoader(testing_set, batch_size=2, shuffle=False, num_workers=0, drop_last=False)
    assert len(testing_set_loader) > 0, "The testing dataset does no contain any samples."

    output_list = []
    filenames_list = []
    for input, filenames in tqdm(testing_set_loader, "Processing"):
        output = model(input)
        output = ((output * testing_set.div) + testing_set.sub).cpu().detach().numpy()
        filenames_list.extend(filenames)
        output_list.append(output)

    return np.vstack(output_list), filenames_list


def train_upscaling(training_set: HSImageDataset, validation_set: HSImageDataset, model: HSUp, loss: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional, epochs: int, batch_size: int, mini_batches: int, valid_batch_size: int, device: str="cuda:0", better_model_cb:Optional[Callable]=None):

    model.to(device)
    training_set_loader = DataLoader(training_set, batch_size=min(len(training_set), batch_size), shuffle=True, num_workers=0, drop_last=False)
    validation_set_loader = DataLoader(validation_set, batch_size=min(len(validation_set), valid_batch_size), shuffle=False, num_workers=0, drop_last=False)

    assert len(training_set_loader) > 0, "The training dataset does no contain any samples. Is the minibatch larger than the amount of samples?"
    assert len(validation_set_loader) > 0, "The validation dataset does no contain any samples. Is the minibatch larger than the amount of samples?"

    bar = tqdm(range(epochs), "Epoch")
    loss_valid = float('inf')
    loss_best = float('inf')
    best_epoch = 0
    best_model = model
    descr = "First epoch"
    for epoch in bar:
        # Train all minibatches
        loss_train = 0
        idx = 0
        if mini_batches is not None:
            minis = range(min(len(training_set_loader), mini_batches))
        else:
            minis = range(len(training_set_loader.dataset))
        training_set_loader_iter = iter(training_set_loader)
        for _ in tqdm(minis, "Batch"):
            bar.set_description(descr)
            bar.refresh()
            input, target = next(training_set_loader_iter)
            optimizer.zero_grad()
            output = model(input)
            ls = loss(output, target)
            ls.backward()
            optimizer.step()
            loss_train += ls.item()
            idx += 1

        if scheduler is not None:
            scheduler.step(epoch)

        # Print stats
        descr = "Epoch {}/{} Loss(T): {:5f} -- Loss(V): {:.5f} -- Loss(Best): {:.5f} @ Epoch {}".format(epoch, epochs, loss_train / idx, loss_valid, loss_best, best_epoch)
        bar.set_description(descr)
        bar.refresh()

        # Validate and save
        bar.set_description("Validating...")
        bar.refresh()
        loss_valid = validate(validation_set_loader, model, loss)
        if loss_valid < loss_best:
            loss_best = loss_valid
            best_epoch = epoch
            best_model = model.clone()
            better_model_cb(model)

    return best_model
