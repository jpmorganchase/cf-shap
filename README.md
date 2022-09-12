<a href="https://www.jpmorgan.com/technology/artificial-intelligence">
<img align="middle" src="./assets/jpmorgan-logo.svg" alt="JPMorgan Logo" height="40">
<img align="middle" src="./assets/xai_coe-logo.png" alt="Explainale AI Center of Excellence Logo" height="75">
</a>

<!-- [![PyPI pyversions](https://img.shields.io/pypi/pyversions/cfshap.svg)](https://pypi.python.org/pypi/cfshap/) -->
<!-- [![PyPI](https://badge.fury.io/py/cfshap.svg)](https://pypi.python.org/pypi/cfshap/) -->
[![License](https://img.shields.io/github/license/jpmorganchase/cf-shap)](https://github.com/jpmorganchase/cf-shap/blob/master/LICENSE)
[![Maintaner](https://img.shields.io/badge/maintainer-Emanuele_Albini-lightgrey)](https://www.emanuelealbini.com)


# Counterfactual SHAP (`cf-shap`)
A modular framework for the generation of counterfactual feature attribution explanations (a.k.a., feature importance). 
This Python package implements the algorithms proposed in the following paper. 
If you use this package please cite our work.

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

**Note that this repository contains the package with the algorithms for the generation of the explanations proposed in the paper and their evaluations _but not_ the expriments themselves.** If you are interested in reproducing the results of the paper, please refer to the [cf-shap-facct22](https://github.com/jpmorganchase/cf-shap-facct22) repository (that uses the algorithms implemented in this repository).
 
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

The following example shows how to use the package to generate the feature importance explanations with 100-NN as counterfactual generator and the SHAP TreeExplainer as feature importance estimator.

```python 

X, y = ... # Must be Numpy arrays
model = ... # Must implement .predict() and .predict_proba() methods.

from emutils.preprocessing.quantiletransformer import EfficientQuantileTransformer
from cfshap.counterfactuals import KNNCounterfactuals
from cfshap.attribution import TreeExplainer, CompositeExplainer
from cfshap.trend import TrendEstimator

MAX_SAMPLES = 10000

X, y = load_dataset(dataset_name, task='classification', return_X_y=True)
model = train_model(X, y, model_type=model_type, params=dict(random_state=random_seed))

# We will need a scaler in the input space for the counterfactual generator
scaler = EfficientQuantileTransformer()
scaler.fit(X)

# Background/Counterfactuals generator
background_generator = KNNCounterfactuals(
    model=model,
    X=X,
    n_neighbors=100,
    distance='cityblock',
    scaler=scaler,
    max_samples=MAX_SAMPLES,
)

# We will need a trend estimator for the attribution estimator
trend_estimator = TrendEstimator(strategy='mean')

# Feature importance estimator
importance_estimator = TreeExplainer(
    model,
    data=None,
    trend_estimator=trend_estimator,
    max_samples=MAX_SAMPLES,
)

# Let's setup the explainer
explainer = CompositeExplainer(
    background_generator,
    importance_estimator,
)

# Let's generate the explanations for the first 10 samples
explanations = explainer(X[:10])
  
```

More documentation to come soon. In the meantime, please contact the authors for any question (see below).


## 3. Contacts and Issues

For further information or queries on this work you can contact the _Explainable AI Center of Excellence at J.P. Morgan_ ([xai.coe@jpmchase.com](mailto:xai.coe@jpmchase.com)) or [Emanuele Albini](https://www.emanuelealbini.com), the main author of the paper.

If you have issues using the package, please open an issue on the GitHub, or contact the authors using the contacts above. We will try to address the issue as soon as possible.

## 4. Disclamer

This repository was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co. and its affiliates (``JP Morgan''), and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation and warranty whatsoever and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful.