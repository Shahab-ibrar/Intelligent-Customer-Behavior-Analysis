import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(path):
    print(f"Running {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    # Configure preprocessor to run cells inside the notebook's parent folder
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path) or '.'}})
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Finished {path} successfully!")

if __name__ == '__main__':
    notebooks = [
        "notebooks/01_data_preprocessing.ipynb",
        "notebooks/02_pca_lda.ipynb",
        "notebooks/03_classification.ipynb",
        "notebooks/04_regression.ipynb",
        "notebooks/05_qlearning_dqn.ipynb"
    ]
    for nb in notebooks:
        try:
            run_notebook(nb)
        except Exception as e:
            print(f"Error running {nb}: {e}")
            raise e
