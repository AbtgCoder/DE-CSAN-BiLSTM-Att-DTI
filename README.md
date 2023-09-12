

# DTI prediction using DE-based CSAN-BiLSTM-Att 

This repository contains code for training and evaluating a DE-based CSAN-BiLSTM-Att model for Drug Target Interaction prediction from scratch. The network is trained and evaluated on Davis and Kiba datasets. Further training and testing has also been performed on BindDB dataset. The final trained models are also present for testing and use.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (version 3.7)
- conda

## Getting Started

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```shell
   git clone https://github.com/AbtgCoder/DE-CSAN-BiLSTM-Att-DTI.git
   ```

2. Navigate to the repository folder:
   ```shell
   cd DE-CSAN-BiLSTM-Att-DTI
   ```

3. Create a virtual enviornment using conda.
   ```shell
   conda create -n your-environment-name python=3.7
   ```

4. switch to newly created enviornment.
   ```shell
   conda activate your-environment-name
   ```

5. Install the dependencies in your enviornment.
   ```shell
   pip install -r requirements.txt
   ```


## Usage

To run the script, use the following command-line arguments:

- `-T` or `--train`: Enable training mode.

Select one of the following datasets:

- `-D` or `--davis`: Select the Davis dataset.
- `-K` or `--kiba`: Select the Kiba dataset.
- `-B` or `--binddb`: Select the BindDB dataset.

If no dataset argument is provided, Kiba dataset is selected by default.


### Examples

#### Training

To enable training on the Davis dataset, run:

```shell
python main.py -T -D
```

To enable training on the Kiba dataset, run:
```shell
python main.py -T -K
```

To enable training on the BindDB dataset, run:
```shell
python main.py -T -B
```

#### Validation
To perform validation without training, omit the -T flag. By default, validation will be done on the Kiba dataset:
```shell
python main.py
```

To enable validation on the Davis dataset, run:

```shell
python main.py -T -D
```

To enable validation on the Kiba dataset, run:
```shell
python main.py -T -K
```

To enable validation on the BindDB dataset, run:
```shell
python main.py -T -B
```

## Model Paths and Hyperparameters
The script sets model paths and hyperparameters based on your dataset selection. Here are the default paths for each dataset:

- Davis dataset: ./final_results/SMILES/Evolutionary/DE/Davis 89.85/Davis_[32, 64, 3, 60, 512, 256, 0.2]-v3.h5
- Kiba dataset: ./final_results/SMILES/Evolutionary/DE/Kiba 97.10/Kiba_[32, 64, 4, 60, 512, 256, 0.2]_smiles-v1.h5
- BindDB dataset: ./final_results/SMILES/Evolutionary/DE/BindDB 84.8/BindDB_(NCFD_32,NCFP_64,CFSD_4,CFSP_4,LSTMdim_60,NFCL_[512, 256],DRFCL_0.2)-v0.h5

## Selected Dataset
The script will display the selected dataset and mode when executed. For example:
```shell
>>> Selected Dataset: Davis
>>> Training Enabled: True
```

## File Descriptions

- `main.py`: The main script to preprocess data, prepare datasets, train and evaluate performance of model with specified hyperparameter configuration and on specified dataset.
- `data_processing.py`: Contains functions for preprocessing the drug and protein data and preparing the training and testing datasets.
- `train_evaluate_csan_bilstm_att_model.py`: Contains functions and code that help in training and evaluating model performance.
- `train_csan_bilstm_att_model.py`: Contains functions for training of model and showcasing evolution of metrics during training.
- `evaluate_csan_bilstm_att_model.py`: Contains functions for evaluation of model and displaying results utilizing various metrics.
- `csan_bilstm_att_model.py`: Contains code for actual implementation of CSAN-BiLSTM-Att model using tensorflow.
- `metrics_csan_bilstm_att_model.py`: Contains code for actual implementation of metrics used for evaluating performance of model.
- `DEoptimization.py`: Contains implementation of DE algorithm. This file can be executed to perform DE optimization for the proposed CSAN-BiLSTM-Att model.

## Results and Evaluation

After training and evaluation, the model's performance will be displayed, including the C-index, MSE, R2m, AUPR and QQ plot. The results will give insights into the model's ability to effectively predict Drug and Protein interaction.

## License

[MIT License](LICENSE.txt)

The project is open source and released under the terms of the MIT License. See the [LICENSE](LICENSE.txt) file for more details.

## Contact

For any questions or inquiries, you can reach me at:
- Email:  [abtgofficial@gmail.com](mailto:abtgofficial@gmail.com)

