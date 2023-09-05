

# DTI prediction using DE-based CSAN-BiLSTM-Att 

This repository contains code for training and evaluating a DE-based CSAN-BiLSTM-Att model for Drug Target Interaction prediction from scratch. The network is trained and evaluated on Davis and Kiba datasets. Further training and testing has also been performed on BindDB dataset. The final trained models are also present for testing and use.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.x)
- NumPy
- Gensim
- Pandas
- deap
- Matplotlib
- scikit-learn

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

3. Train(optional) and Evaluate Model:
   - Run the `main.py` script train the model and evaluate its performance. The DataName , hyperparameter_configuration and MODEL_PATH variables can be modified if only evaluation needs to be performed.
   ```shell
   python main.py
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

