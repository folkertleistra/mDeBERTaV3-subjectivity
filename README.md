# mDeBERTaV3-subjectivity
This GitHub contains all the code and fine-tuned mDeBERTaV3 models that we used for our participation in CLEF2023's CheckThat! Lab (Task 2).

More details about this lab can be found [here](https://checkthat.gitlab.io/).

## Files and Directories

The following files and directories can be found in this repository:
```
├── src
│ ├── make_predictions.py
│ └── mdebertav3_grid.py
├── fine-tuned-models
│ └── models.zip
├── data
│ └── multilingual_adapted.tsv
└── requirements.txt
```

- `src`: Contains the source code files for the project.
  - `make_predictions.py`: Python script that can be used to make predictions using a fine-tuned model on a test set.
  - `mdebertav3_grid.py`: Python script used to run random grid searches using Weights and Biases.

- `fine-tuned-models`: Directory containing a zip file (`models.zip`) that includes all the fine-tuned models.

- `data`: Directory containing the `multilingual_adapted.tsv` file. This file represents the adapted multilingual dataset created by curating and sampling from other datasets available for the task.

- `requirements.txt`: File listing all the required modules and their versions needed to run the scripts in this project.

Additional datasets related to this project can be found in the [CLEF2023-checkthat-lab repository](https://gitlab.com/checkthat_lab/clef2023-checkthat-lab/-/tree/main/task2).

## How to Run

To run the project, follow these steps:

1. Create a Python virtual environment and activate it:

```shell
python3 -m venv myenv
source venv/bin/activate
```

2. Install the required packages from ``requirementst.txt``

```shell
pip install -r requirements.txt
```

3. Now you should be able to run the scripts in the ``src`` directory:

```shell
python3 scrc/make_predictions.py
```

## Acknowledgments

We thank the Center for Information Technology of the University of Groningen for their support and for providing access to the Hábrók high-performance computing cluster.

## Affiliation

This project is affiliated with the University of Groningen.


## Authors

Folkert Atze Leistra

Tommaso Caselli
