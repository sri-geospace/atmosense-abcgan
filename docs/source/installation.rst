About this project
==================
This project uses a generative adversarial network (GAN) to produce a Generator and a Discriminator to characterize the normal atmospheric background. It provides the ability to sample atmospheric parameters, creating full altitude profiles of the sampled measurements. The current system is trained on over 10 years of Inverse Scatter Radar data collected in Alaska at the Poker Flat Radar Range.

Purpose / Vision
================
Currently the project support sampling of low frequency measurements conditioned on high altitude drivers (solar flux, solar zenith angle, etc.). The project goal is to augment this initial capability through generation of high frequency distrubances (waves) as well as allowing conditioning on ground based drivers (terrain, etc.).

Content
=======
The content of this repository entails:

*Readme.md template (this document)
*source code inside 'src/abcgan'
*test code inside 'test'
*tutorials inside 'tutorials'
*pre-trained models inside 'models'

System requirements
===================
This project requires:

*Python 3.8 or newer
*Pytorch 1.8 or newer

Installation
============
This section describes installing this project as a python module.

Basic Installation Steps
------------------
1.Install Dependencies
2.Install Pytorch from the Pytorch page Pytorch installation will be specific to your system configuration depending on gpu availability and drivers.
3.Install Module

Method 1. Install from PyPi
-----------------
This method does not currently work but should be working soon.

*pip install abcgan
Method 2. Install from source
-------------------
This is currently the the preferred method.

1.git clone https://github.com/sri-geospace/atmosense-abcgan.git
2.cd atmosense-abcgan

For end user installation:

3.pip install .
4.Run tests
Make sure to have completed the development. From the top level directory 'atomesense-abcgan' run:

5.python -m unittest

Using the library
=================
This is a library that can be imported directly in python and used. For example usage see 'tutorials/demo.py'.

Contact
=======
For questions please contact Andrew Silberfarb andrew.silberfarb@sri.com