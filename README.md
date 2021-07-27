# About this project

This project uses a generative adversarial network (GAN) to produce a Generator and a Discriminator to characterize the normal atmospheric background. It provides the ability to sample atmospheric parameters, creating full altitude profiles of the sampled measurements. The current system is trained on over 10 yeras of Inverse Scatter Radar data collected in Alaska at the Poker Flat Radar Range.

## Purpose / Vision

Currently the project support sampling of low frequency measurements conditioned on high altitude drivers (solar flux, solar zenith angle, etc.). The project goal is to augment this initial capability through generation of high frequency distrubances (waves) as well as allowing conditioning on ground based drivers (terrain, etc.).

# Content

The content of this repository entails

* Readme.md template (this document)
* source code inside 'src/abcgan'
* test code inside 'test'
* tutorials inside 'tutorials'
* pre-trained models inside 'models'


## System requirements

This project requires:

* `Python 3.8` or newer
* `Pytorch 1.8` or newer

## Installation

This section describes installing this project as a python module.


## Installation

#### Install Dependencies

Install Pytorch from the [Pytorch page](https://pytorch.org/get-started/locally/)
Pytorch installation will be specific to your system configuraiton depending on gpu availability and drivers.

#### Install from PyPi
This method does not currently work but should be working soon.
```bash
pip install abcgan
```

#### Install from source
This is currently the the preferred method.
```bash
git clone https://github.com/sri-geospace/atmosense-abcgan.git
```

```cmd
cd atmosense-abcgan
```
For end user installation:
```cmd
pip install .
```

### Run tests

Make sure to have completed the development. From the top level directory 'atomesense-abcgan' run:

```bash
python -m unittest
```

# Using the library

This is a library that can be imported directly in python and used. For example usage see 'tutorials/demo.py'.

# Contact

For questions please contact Andrew Silberfarb <andrew.silberfarb@sri.com>
