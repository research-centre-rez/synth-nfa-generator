Synthetic NFA Generator
---


# Installation

Follow these instruction to install editable version of this code. 

Clone the repo and enter the directory `synth-nfa-generator`. Then activate virtual environment as in example below

```bash
python -m venv .venv
.venv/bin/activate
```

Install the code as editable

```bash
pip install -e .
```

In case you want to use the notebooks then additionally install `jupytext`, `jupyter-lab`, `ipykernel`.

```bash
pip install jupytext jupyter-lab ipykernel
```

Use jupytext to convert all notebooks in the format `.md` to `.ipynb`.

```bash
jupytext --to ipynb notebook/*.md
```

To make the virtual env visible from jupyter use:
```
python -m ipykernel install --user --name 'synth'
```

And run `jupyter lab` command.


To make sure that mitsuba runs as expected, try running `minimal.py`
