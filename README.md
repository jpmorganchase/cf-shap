<a href="https://www.jpmorgan.com/technology/artificial-intelligence">
<img align="middle" src="./assets/jpmorgan-logo.svg" alt="JPMorgan Logo" height="40">
<img align="middle" src="./assets/xai_coe-logo.png" alt="Explainale AI Center of Excellence Logo" height="75">
</a>

<!-- [![PyPI pyversions](https://img.shields.io/pypi/pyversions/cfshap.svg)](https://pypi.python.org/pypi/cfshap/) -->
<!-- [![PyPI](https://badge.fury.io/py/cfshap.svg)](https://pypi.python.org/pypi/cfshap/) -->
[![License](https://img.shields.io/github/license/jpmorganchase/cf-shap)](https://github.com/jpmorganchase/cf-shap/blob/master/LICENSE)
[![Maintaner](https://img.shields.io/badge/maintainer-Emanuele_Albini-lightgrey)](https://www.emanuelealbini.com)


# Counterfactual SHAP (`cf-shap`)
A modular framework for the generation of counterfactual feature attribution explanations (a.k.a., feature importance). This Python package implements the algorithms proposed in the following paper. If you use this package please cite our work.

**Counterfactual Shapley Additive Explanations**  
Emanuele Albini, Jason Long, Danial Dervovic and Daniele Magazzeni  
J.P. Morgan AI Research  
[ACM](https://dl.acm.org/doi/abs/10.1145/3531146.3533168) | [ArXiv](https://arxiv.org/abs/2110.14270)

```
@inproceedings{Albini2022,
  title = {Counterfactual {{Shapley Additive Explanations}}},
  booktitle = {2022 {{ACM Conference}} on {{Fairness}}, {{Accountability}}, and {{Transparency}}},
  author = {Albini, Emanuele and Long, Jason and Dervovic, Danial and Magazzeni, Daniele},
  year = {2022},
  series = {{{FAccT}} '22},
  pages = {1054--1070},
  doi = {10.1145/3531146.3533168}
}
```

**Note that this repository contains the package with the algorithms for the generation of the explanations proposed in the paper and their evaluations BUT NOT the expriments themselves.** If you are interested in reproducing the results of the paper, please refer to the [cf-shap-facct22](https://github.com/jpmorganchase/cf-shap-facct22) repository (that uses the algorithms implemented by this package).
 
## 1. Installation
To install the package manually, simply use the following commands. Note that this package depends on shap>=0.39.0 package: you may want to install this package or the other dependencies manually (using conda or pip). See setup.py for more details on the dependencies of the package.

```bash
# Clone the repo into the `cf-shap` directory
git clone https://github.com/jpmorganchase/cf-shap.git

# Install the package in editable mode
pip install -e cf-shap
```
NOTE: You may want to install the dependencies manually, check `install_requires` in `setup.py` for the list of dependencies for the package.

## 2. Basic Usage Example

```python 

X_train, X_test = ... # Must be Numpy arrays
model = ... # Must implement .predict() and .predict_proba() methods.

from emutils.preprocessing.quantiletransformer import EfficientQuantileTransformer
from cfshap.counterfactuals import KNNCounterfactuals
from cfshap.attribution import TreeExplainer, CompositeExplainer
from cfshap.trend import TrendEstimator

MAX_SAMPLES = 10000

scaler = EfficientQuantileTransformer()
scaler.fit(X_train)

trend_estimator = TrendEstimator(strategy='mean')

explainer = CompositeExplainer(
  TreeExplainer(
      model,
      data=None,
      feature_perturbation='interventional',
      trend_estimator=trend_estimator,
      max_samples=MAXSAMPLES,
  ),
  KNNCounterfactuals(
      model=model,
      X=X_train,
      n_neighbors=k,
      distance='cityblock',
      scaler=scaler,
      max_samples=MAX_SAMPLES,
  )
)
```

More documentation to come soon. In the meantime, please contact the authors for any question (see below).


## 3. Contacts

For further information or queries on this work you can contact the _Explainable AI Center of Excellence at J.P. Morgan_ ([xai.coe@jpmchase.com](mailto:xai.coe@jpmchase.com)) or [Emanuele Albini](https://www.emanuelealbini.com), the main author of the paper.

## 4. Disclamer

This repository was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co. and its affiliates (``JP Morgan''), and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation and warranty whatsoever and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation,
offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful.