# ABCGAN

This project uses a generative adversarial network (GAN) to produce a Generator and a Discriminator to characterize the normal atmospheric background. It provides the ability to sample atmospheric parameters, creating full altitude profiles of the sampled measurements. The current system is trained on over 10 years of Incoherent Scatter Radar data collected in Alaska at the Poker Flat Research Range.

Currently the project supports the sampling of low frequency measurements conditioned on high altitude drivers (solar flux, solar zenith angle, etc.). The project goal is to augment this initial capability through generation of high frequency distrubances (waves) as well as allowing conditioning on ground based drivers (terrain, etc.).

## Installing abcgan

This package is available on PyPI and can be installed with pip:

```bash
pip install abcgan
```

It requires Python 3.8+ and uses PyTorch 1.8+.

## Contents

The content of this repository entails:

* Source code inside 'src/abcgan'
* Test code inside 'test'
* Tutorials inside 'tutorials'
* Pre-trained models inside 'models'

## Installation from source

Install Pytorch from the [Pytorch page](https://pytorch.org/get-started/locally/)
Pytorch installation will be specific to your system configuraiton depending on gpu availability and drivers.

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

## Building docs

Documents are available on [Read the Docs](https://github.com/valentic/atmosense-abcgan). You may also create the docs. Configuration files are supplied, including a requirements file with the necessary dependencies. 

```cmd
cd docs
pip install -r requirements.txt
make html
```

generate source files using api-doc (see https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html for details) and return to docs root
```cmd
sphinx-apidoc -o . ../src/abcgan
```
note that .rst files are generated from the installed module not the source tree; to reference local changes make sure the installation is performed with pip install -e .

add the lines to conf.py if not there already
```cmd 
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   abcgan
   modules
```
finally, build the docs
```cmd
cd ..
make clean
```
```cmd
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
