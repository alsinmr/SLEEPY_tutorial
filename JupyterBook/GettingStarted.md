# <font color="maroon">Getting Started</font>

There are currently two possiblities for running SLEEPY: with a local Python installation, or online via [Google Colab](https://colab.research.google.com/). A local installation of SLEEPY may be run in [Jupyter notebooks](https://jupyter.org/), as a script, or in a Python/iPython terminal. Here we provide a few notes for a local installation or for running in Google Colab.


## Local Installation
SLEEPY does not currently exist for installation via [pip](https://pypi.org/) or [conda](https://docs.conda.io/en/latest/), and instead is downloaded from GitHub, with its containing folder added to the system path (or placed with other Python modules). The required modules must be installed separately by the user.

[**SLEEPY on GitHub**](https://github.com/alsinmr/SLEEPY/)

Versions listed tested for pyDR, although most recent module versions should work.

### Requirements
* Python 3 (3.7.3)
* numpy (1.19.2)
* matplotlib (3.4.2) 
* scipy (1.5.2)

### Recommended Installations
* [Jupyter notebooks](https://jupyter.org/): Neat code organization based in a web browser

If you're starting from scratch, you may consider installing [Anaconda](https://anaconda.org), which will install Python 3, numpy, matplotlib, scipy, and Jupyter Notebooks for you. The remaining packages (MDAnalysis, etc.) may then be installed with conda (or less ideally, with pip). ChimeraX is not installed with Python, but has its own installer (you will need to tell pyDR where ChimeraX is installed, use pyDR.chimeraX.chimeraX_funs.set_chimera_path).

Once set up, any of the webpages with code in the tutorial can be downloaded with the button in the upper right (pick .ipynb) and run locally. 

## Google Colab
Colab has the advantage that there is no local installation requirement, with the base requirements for Python already setup. The notebooks that we provide include a few lines at the beginning that install the additional requirements for MDAnalysis and NGL viewer, so you don't have to know how to set it up– just execute the cells (shift+enter executes a cell).

For all notebooks in the tutorial, there is a button near the top that redirects you to Google Colab, where you can then edit and execute contents of the notebook: ![](https://colab.research.google.com/assets/colab-badge.svg)