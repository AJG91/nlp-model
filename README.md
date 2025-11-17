# Natural language processing (NLP) model

[my-website]: https://AJG91.github.io "my-website"

This repository contains code that demonstrates how transformers process text by building a DistilBERT sentiment classifier (NLP model).

## Getting Started

* This project relies on `python=3.12`. It was not tested with different versions
* Clone the repository to your local machine
* Once you have, `cd` into this repo and create the virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate nlp-model-env`
* Install the packages in the repo root directory using `pip install -e .` (you only need the `-e` option if you intend to edit the source code in `nlp_model/`)

## Example

See [my website][my-website] for examples on how to use this code.

## Citation

If you use this project, please use the citation information provided by GitHub via the **“Cite this repository”** button or cite it as follows:

```bibtex
@software{Garcia2025NLPModel,
  author = {Alberto J. Garcia},
  title = {NLP Model},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/AJG91/nlp-model},
  license = {MIT}
}
```