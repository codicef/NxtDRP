# NxtDRP  
In this project, we present NxtDRP, a method that predicts drug responses in cancer cell lines by integrating multi-omics data. Using the NXTfusion library, NxtDRP combines RNA-Seq expression and Proteomics profiles within Entity-Relation graphs to predict DRP target such as IC50 and Aread Under Dose-Response Curve (AUDRC). We apply NxtDRP to the GDSC dataset, exploring various train-test splits and prediction strategies. This method utilizes multi-omics data for drug response predictions, enriching the analysis of cancer cell lines.

## Setting Up the Conda Environment
To create and activate the Conda environment for this project, follow these steps:

1. **Navigate to the Project Directory**: Open a terminal and change to the directory where the `environment.yml` file is located.

2. **Create the Conda Environment**: Run the following command to create a Conda environment based on the specifications in `environment.yml`:
    ```bash
    conda env create -f environment.yml
    ```

    This command reads the `environment.yml` file in your current directory, creating a new Conda environment with all the specified packages and their versions.

3. **Activate the Conda Environment**: Once the environment is created, you can activate it using:

    ```bash
    conda activate nxtdrp
    ```


### Note

If you update the project's dependencies, you can update the Conda environment to reflect the changes in `environment.yml` by running:

```bash
conda env update -f environment.yml --prune
```


## Running the Predictor

1. **Prepare Your Dataset**: Make sure your dataset is in the correct format and located in the specified directory (`./data/datasets/`). 

To prepare the dataset, use the following command:
```bash
python src/data.py 
```

2. **Run the Predictor**: Use the following command to start the predictive modeling process. Adjust the parameters as needed:

```bash
python src/main.py --model NxtDRP --n_tests 40 --cv_type random_split --default_hp True --device cuda
```

### Parameters:
- `--model`: Choose between `NxtDRP` and `NxtDRPMC` models.
- `--n_tests`: Number of randomized tests to perform.
- `--cv_type`: Type of cross-validation strategy. Options are `random_split`, `unseen_cell`, and `unseen_drug`.
- `--default_hp`: Use default hyperparameters (`True`) or optimize them (`False`).
- `--device`: Computation device, `cuda` for GPU or `cpu`.


## License 
This project is licensed under the MIT License.
