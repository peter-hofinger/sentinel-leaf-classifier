Leaf Type Mixture
==============================

> Disclaimer: This repository is a work in progress and the corresponding paper has not been published yet. The code is subject to change.

Welcome to the repository for the paper "Predicting Leaf Type Mixture Using Tailored Sentinel-2 Composites." Our goal is to ensure all results are reproducible. If you encounter any issues, please open an issue in this repository.

# Overview

In this study, we investigate the potential of Sentinel-2 Level-2A data for predicting leaf type mixtures of forests in temperate climates. We focus on optimizing spectral band combinations, compositing methods, and hyperparameters for various regression models to enhance Sentinel-2 Level-2A data for predicting leaf type mixtures. The experiments are broken down into:

1. **Band Importance:** We conduct a systematic evaluation using recursive feature elimination (RFE) with a random forest model to identify significant bands and indices.
2. **Compositing:** The best number of composites per compositing method and year is determined to refine the dataset.
3. **Hyperparameter Tuning:** Hyperparameters for multiple regression models are optimized and the best model is selected based on performance.
4. **Generalization:** We evaluate the best model's generalization capability across unseen temporal windows and experimental sites.

Our results highlight the potential of tailored Sentinel-2 composites for leaf type mixture predictions and provide insights into optimizing spectral band combinations and compositing methods.

# Installation and Setup

This repository requires **Python 3.10** or later. To install the required packages, run the following command in the root directory of this repository:

```bash
pip install -e .
```

To use the **Earth Engine API**, authenticate with the following Python code. A browser tab should open after you execute the code. Follow the website instructions and paste the authentication key into the input window. Repeat this process whenever your token expires.

```python
import ee

ee.Authenticate()
```

# Usage

All experiments described in our paper are implemented in the notebooks within the `notebooks` directory. After completing the installation and setup, you can run these notebooks in your preferred IDE.

# CI/CD

This GitHub repository uses [GitHub Actions](https://github.com/features/actions) for continuous integration (CI). The CI pipeline is defined in the `.github/workflows` directory. It includes:
- **Code quality checks** using [pylint](https://pylint.readthedocs.io/)
- **Testing** using [pytest](https://docs.pytest.org/).

> *Functions using the Earth Engine API are not tested due to authentication concerns.*

To run the CI workflow locally, execute the following commands from the repository directory:
```bash
pip install -e .[linting,testing]
pylint --disable=line-too-long,too-many-lines,no-member ltm
pylint --disable=line-too-long,too-many-lines,no-member,missing-module-docstring,missing-class-docstring,missing-function-docstring test
pytest
```

The code was formatted using the following commands:
```bash
pip install isort black[jupyter]
isort --profile black .
black .
```

# Acknowledgements

We would like to thank all the contributors and reviewers for their valuable feedback and support. This work was supported by [Funding Source].

# Contact

For questions or contributions, please contact Peter Hofinger at [hofinger-peter@gmx.de](hofinger-peter@gmx.de).

# Citation

If you use this repository in your research, please cite our paper:

```
@article{paper2024,
  title={Paper Title},
  author={Name and Coauthor Names},
  journal={Journal Name},
  year={2024},
  volume={xx},
  number={xx},
  pages={xx--xx},
  doi={xx.xxxx/xxxxxx},
}
```

Thank you for using our code and contributing to the advancement of science!
