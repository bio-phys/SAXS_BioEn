# SAXS_BioEn

Using a simple example, the Jupyter notebook SAXS\_BioEn\_example.ipynb shows how
to refine structural ensembles with BioEn (https://github.com/bio-phys/bioen)
and using SAXS data. At the end of the notebook and as an option, we evaluate
the h-statistic and (h,$\chi^2$)-statistic provided by the hplusminus software (https://github.com/bio-phys/hplusminus) to validate
the goodness-of-refinement. 

The Jupyter notebook runs using Python 3 and the current versions of the bioen and hplusminus software.

## Installation

The notebook requires Python >= 3.9, bioen >= 0.1.4, and hplusminus >= 1.0.2,
which work with both NumPy 1.x and NumPy 2.x. All dependencies can be installed from PyPI, e.g.
into a fresh virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook SAXS_BioEn_example.ipynb
```

Binary wheels of bioen are available for Linux (x86_64, glibc >= 2.28). On other platforms,
pip builds bioen from source, which requires a C compiler and the GSL library,
see https://github.com/bio-phys/bioen for details. 

**Try online without installation!**

You can try out the example **online and without installation** using Binder:

https://bio-phys.pages.mpcdf.de/SAXS_BioEn/

