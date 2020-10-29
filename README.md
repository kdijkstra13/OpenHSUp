#Introduction

The HSUp (HyperSpectral Upscaling) repository contains a deep learning algorithm to perform hyperspectral upscaling and demosaicking of 
multi-color-filter arrays (MCFA). This repository is the reference implementation of the paper: 
"Hyperspectral demosaicking and crosstalk correction using deep learning"

The original code has been ported from Caffe to PyTorch.

p.s.
The crosstalk correction has not been ported yet.  

#Usage
- The file **./train.py** can be executed with command line parameters.
- Run **./train_hsup.py** for a demonstration:
    1. Images in ./data/4x4/training are used for training.
    2. Images in ./data/4x4/validation are used for validation.    
    3. results are placed in the ./data/output/ folder).
    4. SSIM is calculated for the upscaled result.

Result of the upscaling (mapped from 16-channel to RGB)\
<img src="./data/output/example_upscale.jpg" alt="upscale" width="200px"/>

Result of the demosaicking (mapped from 16-channel to RGB)\
<img src="./data/output/example_demosaick.jpg" alt="upscale" width="200px"/>

Note:
the images appear *very* dark in a normal .png viewing tool because 
the images are displayed as 16 bit while the signal from the camera is only 10 bit.

#Adding other mosaic patterns
This repository is tested for a 4x4 mosaic.
If other mosaic sizes need be handled please update the HSUp class in models.py.

#Citing OpenHSUp
If this code benefits your research please cite:

@article{dijkstra2019hyperspectral,\
&nbsp;&nbsp;title={Hyperspectral demosaicking and crosstalk correction using deep learning},\
&nbsp;&nbsp;author={Dijkstra, Klaas and van de Loosdrecht, Jaap and Schomaker, Lambert RB and Wiering, Marco A},\
&nbsp;&nbsp;journal={Machine Vision and Applications},\
&nbsp;&nbsp;volume={30},\
&nbsp;&nbsp;number={1},\
&nbsp;&nbsp;pages={1--21},\
&nbsp;&nbsp;year={2019},\
&nbsp;&nbsp;publisher={Springer}\
}

#Copyright notice
Copyright (C) 2020 Klaas Dijkstra

OpenHSUp is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.