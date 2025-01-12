# <font color="#B00000">Getting Started</font>

There are currently two possiblities for running SLEEPY: with a local Python installation, or online via [Google Colab](https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabTemplate.ipynb). A local installation of SLEEPY may be run in [Jupyter notebooks](https://jupyter.org/), as a script, or in a Python/iPython terminal. Here we provide a few notes for a local installation or for running in Google Colab.


## Local Installation
SLEEPY does not currently exist for installation via [pip](https://pypi.org/) or [conda](https://docs.conda.io/en/latest/), and instead is downloaded from GitHub, with its containing folder added to the system path (or placed in a folder with other installed Python modules). The required modules must be installed separately by the user.

[**SLEEPY on GitHub**](https://github.com/alsinmr/SLEEPY/)

Versions listed have been tested for SLEEPY, although any recent module version should work.

### Requirements
* Python 3 (3.7.3)
* numpy (1.19.2)
* matplotlib (3.4.2) 
* scipy (1.5.2)

### Recommended Installations
* [Jupyter notebooks](https://jupyter.org/): Neat code organization based in a web browser

If you're starting from scratch, you may consider installing [Anaconda](https://anaconda.org), which will install Python 3, numpy, matplotlib, scipy, and Jupyter Notebooks for you. 

Once set up, any of the webpages with code in the tutorial can be downloaded with the button in the upper right (pick .ipynb) and run locally. 

## Google Colab
Colab has the advantage that there is no local installation requirement, with the basic Python modules already setup. The notebooks that we provide include a few lines at the beginning that download SLEEPY and import it.

For all notebooks in the tutorial, there is a button near the top that redirects you to Google Colab, where you can then edit and execute contents of the notebook: <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabTemplate.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>