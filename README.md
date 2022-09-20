# uu-methods-ai-research
This repo contains the assignments made for the Methods in A.I. Research Course, Utrecht University, 1st Period.

## Build Dependencies
- This project uses [pyenv](https://github.com/pyenv/pyenv) in conjunction with [poetry](https://python-poetry.org/) for better package management. 
Assuming both are **properly installed** and you're **inside the directory** where the repository has been cloned, run `poetry install` and `poetry shell` to install all dependencies and setup a local virtualenv with said packages, respectively. `poetry run main.py` will run the main file inside the virtualenv.

## Arguments
- Two arguments are supported when running `main.py`: `reprocess`, which forces the program to rebuild and reprocess the dataframe from the original dataset (dialog_acts.dat) and `remodel`, which forces the remodelling of the current models selected.

### Deadlines:

- 1a. Text Classification - Sept 16th

- 1b. Dialog Management - Sept 23th

- 1c. Reasoning and Configurability - Sept 30th

- **1d. Final Submission - October 9th**
