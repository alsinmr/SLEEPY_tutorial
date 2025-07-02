# <font color="#B00000">Getting Started</font>

There are currently three possiblities for running SLEEPY: with a local Python installation, online via [Google Colab](https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabTemplate.ipynb), or online via [myBinder.org](https://mybinder.org/v2/gh/alsinmr/SLEEPY_tutorial/main). A local installation of SLEEPY may be run in [Jupyter notebooks](https://jupyter.org/), as a script, or in a Python/iPython terminal. Here we provide a few notes for a local installation or for running in Google Colab.
 

## Local Installation
The stable version of SLEEPY can be installed with pip (or alternatively downloaded from [PyPI](https://pypi.org/project/sleepy-nmr/)):
```
pip install sleepy-nmr
```
Note that the resulting module is called SLEEPY, as opposed to sleepy-nmr. Import as follows (shortening name to 'sl' is optional)
```
import SLEEPY as sl
```

SLEEPY does not exist for installation via [conda](https://docs.conda.io/en/latest/), but package requirements for SLEEPY are relatively limited (numpy, scipy, matplotlib), so installing via pip is unlikely to break your conda installation. 

For the latest (potentially less stable) version of SLEEPY, download from GitHub and add its folder to your system path. The required modules must be installed separately by the user.

[**SLEEPY on GitHub**](https://github.com/alsinmr/SLEEPY/)

Versions listed have been used for [benchmarking SLEEPY](Chapter7/Ch7_SleepyBenchmark), although we expect most recent versions to be stable.

### Requirements
* Python 3 (3.11.13)
* numpy (1.24.3)
* matplotlib (3.4.2) 
* scipy (1.5.2)

### Recommended Installations
* multiprocess (0.70.15 – in benchmarking, much faster than the built-in multiprocessing)
* [Jupyter notebooks](https://jupyter.org/): Neat code organization based in a web browser
* ipywidgets (7.7.1): Used primarily in Google Colab to make zoomable plots

Note that we have used numpy and scipy with Intel's MKL BLAS/LAPACK libraries. In our tests, this results in considerable speedup. Installation can be challenging. 
* Currently will not install with python>3.11
* May not yield improvements if not using an Intel processor (e.g. Apple Silicon, esp. AMD)
* We used conda for installation
conda config --add channels defaults
conda config --set channel_priority strict
conda install numpy scipy mkl mkl-service libblas=*=*mkl


If you're starting from scratch, you may consider installing [Anaconda](https://anaconda.org), which will install Python 3, numpy, matplotlib, scipy, and Jupyter Notebooks for you. This *should* also use the MKL libraries.

Once set up, any of the webpages with code in the tutorial can be downloaded with the button in the upper right (pick .ipynb) and run locally. 

## Google Colab
Colab has the advantage that there is no local installation requirement, with the basic Python modules already setup. The notebooks that we provide include a few lines at the beginning that download SLEEPY and import it.

For all notebooks in the tutorial, there is a button near the top that redirects you to Google Colab, where you can then edit and execute contents of the notebook: <a href="https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabTemplate.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg"></a>