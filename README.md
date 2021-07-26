# About this project

This abcgan project provides the machine learning tools needed for the Atmosense program. Specifically this project will use a generative adversarial network (GAN) that produces both a Generator and a Discriminator to characterize the normal atmospheric background. The project will be in two phases the first phase focusing no low frequency background variables (BVs) and the second focusing on high frequency perturbations (HFPs).

## Purpose / Vision

This project is intended for SRI internal development of the Generator and Discriminator for the Atmosense program only. After development and training versions will be released to GitHub for public consumption and use. We will tag and branch for each release of the generator / discriminator on GitHub. Additionally we will maintain documentation on Gitlab pages, and demonstration scripts for install and use of both the generator and discriminator in the tutorials folder.

The project will characterize the normal atmospheric background in two ways, using a Generator that can generate samples of the background, and a discriminator that can distinguish normal background samples from fake or unusual measurements. The both the generator and the discriminator will input a set of driving parameters describing the external environment (time of day, solar driving, etc.). The generator will output a list of correlated measurements across multiple altitude bins. The discriminator will input both the driving parameters and a list of measurements and output a list of scores, one for each altitude, indicating the agreement of that altitudes measurement with the normally observed background.

# Content

The content of this repository entails

* Readme.md template (this document)
* source code inside 'src/abcgan'
* test code inside 'test'
* documentation
* python module definition template
  - setup.py
  - MANIFEST.in the manifest allows adding non-python module specific components to the delivery (e.g. documentation, changelog, network parameters, etc.)
* general meta documents
  - CHANGELOG
  - LICENSE
* SRI meta information template (this should not be given to a client)
* docker configuration (build and docker-compose)
* .gitignore


## System requirements

This project requires:

* `Python 3.6` or newer
* `Pytorch 1.7` or newer

## Installation

This section describes installing this project as a python module.


## Installation

#### Install Dependencies

Install Pytorch from the [Pytorch page](https://pytorch.org/get-started/locally/)

#### Install from source
This is the preferred method.
```bash
git clone https://gitlab.sri.com/Atmosense/abcgan.git
```

```cmd
cd abcgan
```
For end user installation:
```cmd
pip install .
```

Alternatively, for development, use symbolic links to support dynamic reloading:
```cmd
pip install -e .

### Run tests

Make sure to have completed the development.

```bash
python setup.py test
coverage run --source="src" -m unittest discover -s tests/
coverage report
coverage html
```

Alternatively you may also run the unit tests with: `python -m unittest discover -s tests`

# Build Docker image

To build the docker image, you need to have [Docker](https://www.docker.com/) and [Docker-Compose](https://docs.docker.com/compose/install/) installed and run the following command in the project directory:

```bash
docker build -t pythontemplate .
```

You may run the application with:

```bash
docker run pythontemplate
```

## Docker Compose

Docker compose let's you run multiple images in combination and define some configurations outside the docker image for ease of configuration and use. To start this image via docker-compose use the command:

```bash
docker-compose up
```

To shutdown enter:

```bash
docker-compose down
```

# Run the application

There is no external application. The code is intended to be used as a library, or in an interactive session. Examples of use are contained in the examples folder (not yet present).

# Contact

For questions please contact Andrew Silberfarb <andrew.silberfarb@sri.com>
