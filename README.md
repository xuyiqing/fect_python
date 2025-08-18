# fect_python
<!-- badges: start -->

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- badges: end -->

Fixed effects counterfactual estimator (**fect**) based on **rpy2**. This is a package for implementing counterfactual estimators in panel
fixed-effect settings. It is suitable for panel/TSCS analysis with
binary treatments under (hypothetically) baseline randomization. It
allows a treatment to switch on and off and limited carryover effects.
It supports linear factor models—hence, a generalization of
[**gsynth**](https://yiqingxu.org/packages/gsynth/index.html)—and the
matrix completion method.

**Repo:** [GitHub](https://github.com/xuyiqing/fect) (1.0.0)

**Examples:** The original R
[tutorial](https://yiqingxu.org/packages/fect/articles/tutorial.html) can be replicated in Python with [this](https://github.com/xuyiqing/fect_python/blob/main/example/fect_ipy.ipynb) tutorial. You can also find a markdown version tutorial [here](https://github.com/xuyiqing/fect_python/blob/main/fect_python_totorial_md/README.md)

**Reference:** Licheng Liu, Ye Wang, Yiqing Xu (2021). [A Practical
Guide to Counterfactual Estimators for Causal Inference with Time-Series
Cross-Sectional
Data](https://yiqingxu.org/papers/english/2022_fect/LWX2022.pdf).
*American Journal of Political Science*, conditionally accepted.

The original **R** package of **fect** can be found here: [GitHub](https://github.com/xuyiqing/fect) (1.0.0).
This package is based on [**rpy2**](https://rpy2.github.io/), which makes R objects available in Python environments.

## Installation
You can install **fect** from github by the following command:
```
pip install git+https://github.com/xuyiqing/fect_python.git
```

## Requirements
- Python 3.7+
- rpy2 3.5+
- numpy 1.1+
- pandas 1.1.2+

Although **fect_python** works on the latest version of **rpy2**, we strongly recommend installing **rpy2 3.5.4** in case of potential conflits.
