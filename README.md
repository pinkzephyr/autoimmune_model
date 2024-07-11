# Simulating T cell thymic selection and persistent/major infection

This repository contains the code for the paper "How persistent infection overcomes peripheral tolerance mechanisms to cause T cell–mediated autoimmune disease". Specifically, it contains the code for modeling T cell development and persistent infection that may lead to autoimmunity. The peptide_files.zip contains csv files of foreign IAb peptides with corresponding IAb mouse peptide homologs. The title of each file is the Mycobacterium tuberculosis IAb-binding 15mer peptide followed by its 9mer core, and in the csv file contains the corresponding IAb 15mer mouse peptide and its 9mer core.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/pinkzephyr/autoimmune_model.git
    cd autoimmune_model
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv model
    source model/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

    Or with Conda:
    ```bash
    conda env create -f environment.yml
    conda activate env-name
    ```

## Usage

To run the main script:
```bash
python src/model.py
