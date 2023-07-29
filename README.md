# ABCGAN

This project uses a generative adversarial network (GAN) to produce a Generator and a Discriminator to characterize the normal atmospheric background. It provides the ability to sample atmospheric parameters, creating full altitude profiles of the sampled measurements. The current system is trained on over 10 years of Incoherent Scatter Radar data collected in Alaska at the Poker Flat Research Range.

Currently the project supports the sampling of low frequency measurements conditioned on high altitude drivers (solar flux, solar zenith angle, etc.). The project goal is to augment this initial capability through generation of high frequency distrubances (waves) as well as allowing conditioning on ground based drivers (terrain, etc.).

[![Documentation Status](https://readthedocs.org/projects/atmosense-abcgan/badge/?version=latest)](https://atmosense-abcgan.readthedocs.io/en/latest/?badge=latest)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sri-geospace/atmosense-abcgan/HEAD?labpath=tutorials%2Fdemo.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5889628.svg)](https://doi.org/10.5281/zenodo.5889628)

## Installing abcgan

This package is available on PyPI and can be installed with pip. It is
best to do this in a Python virtual environment:

```bash
python -m venv venv
. venv/bin/activate

pip install abcgan
```

It requires Python 3.8+ and uses PyTorch 1.8+.

## Downloading the tutorials

The tutorials are available as Jupyter notebooks and are located along
with sample data in the tutorial directory in the github repository.
You can download these tutorials with the abcgan-cmd program which is
installed along with the abcgan package.

After downloading the tutorials, install the required packages listed
in tutorials/requirements.txt using pip.  You can then start Jupyter Lab
and load the tutorial notebook tutorial/demo.ipynb.

The data files and models used in the tutorials are approximately 600Mb
in size. Use the -m flag to abcgan-cmd to download them with the tutorials.

```cmd
abcgan-cmd -m download tutorials
pip install -r tutorials/requirements.txt

jupyter lab 
```
## Running the tutorials on Binder

You can run the tutorial notebooks at any time on [Binder](https://mybinder.org). A Docker container will be automatically created with the abcgan package and tutorials.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sri-geospace/atmosense-abcgan/HEAD?labpath=tutorials%2Fdemo.ipynb)

https://mybinder.org/v2/gh/sri-geospace/atmosense-abcgan/HEAD?labpath=tutorials%2Fdemo.ipynb

## Contents

The content of this repository entails:

* Source code inside 'src/abcgan'
* Test code inside 'test'
* Tutorials inside 'tutorials'
* Pre-trained models inside 'models'
* Helper scripts in 'contrib'

## Installation from source

Install Pytorch from the [Pytorch page](https://pytorch.org/get-started/locally/)
Pytorch installation will be specific to your system configuraiton depending on gpu availability and drivers.

```bash
git clone https://github.com/sri-geospace/atmosense-abcgan.git
```

For end user installation:
```cmd
pip install atmosense-abcgan 
```

For development:
```cmd
pip install -e atmosense-abcgan 
```

## Building the documentation 

The package documentation is available on [Read the Docs](https://github.com/valentic/atmosense-abcgan). You may also build the docs locatlly. The configuration files and document sources are in the docs/ directory, including a requirements file with the necessary dependencies. 

```cmd
cd docs
pip install -r requirements.txt
make html
```

The API source files are generated using [api-doc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html):
```cmd
sphinx-apidoc -o . ../src/abcgan
```
Note that .rst files are generated from the installed module not the source tree; to reference local changes make sure the installation is performed with pip install -e .

Add the lines to conf.py if not there already:
```cmd 
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   abcgan
   modules
```
Finally, build the docs:
```cmd
make clean
make html
```

## Run tests

Make sure to have completed the development. From the top level directory 'atomesense-abcgan' run:

```bash
python -m unittest
```

## Using the library

This is a library that can be imported directly in python and used. For example usage see 'tutorials/demo.py'.

## Contact

For questions please contact Andrew Silberfarb <andrew.silberfarb@sri.com>
