<a href="https://www.jpmorgan.com/technology/artificial-intelligence">
<img align="middle" src="./assets/jpmorgan-logo.svg" alt="JPMorgan Logo" height="40">
<img align="middle" src="./assets/xai_coe-logo.png" alt="Explainale AI Center of Excellence Logo" height="75">
</a>


# Counterfactual SHAP (`cf-shap`)
A modular framework for the generation of counterfactual feature attribution explanations (a.k.a., feature importance). This Python package implements the algorithms proposed in the following paper. If you use this package please cite our work.

**Counterfactual Shapley Additive Explanations**  
Emanuele Albini, Jason Long, Danial Dervovic and Daniele Magazzeni  
J.P. Morgan AI Research  
[ACM](https://dl-acm-org.iclibezp1.cc.ic.ac.uk/doi/abs/10.1145/3531146.3533168) | [ArXiv](https://arxiv.org/abs/2110.14270)

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
 
## 1. Installation
To install the package manually, simply use the following commands. Note that this package depends on shap>=0.39.0 package: you may want to install this package or the other dependencies manually (using conda or pip). See setup.py for more details on the dependencies of the package.

```bash
# Clone the repo into the `cf-shap` directory
git clone https://github.com/jpmorganchase/cf-shap.git

# Install the package in editable mode
pip install -e cf-shap
```


## 2. Usage

```python 
from cfshap import ...

TODO

```


## 3. Contacts

For further information or queries on this work you can contact the _Explainable AI Center of Excellence at J.P. Morgan_ ([xai.coe@jpmchase.com](mailto:xai.coe@jpmchase.com)) or [Emanuele Albini](https://www.emanuelealbini.com), the main author of the paper.
