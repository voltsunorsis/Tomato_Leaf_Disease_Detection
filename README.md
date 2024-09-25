# Tomato Disease Classification Project

This project uses deep learning to classify tomato plant diseases based on leaf images. It includes model training, evaluation, and a Flask web application for making predictions.

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- Flask
- Pillow
- scikit-learn
- matplotlib
- seaborn
- numpy

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install torch torchvision Flask Pillow scikit-learn matplotlib seaborn numpy
   ```

3. Prepare your dataset:
   - Organize your tomato leaf images in the following structure:
     ```
     Tomato/
     ├── Train/
     │   ├── Bacterial_Spot/
     │   ├── Early_Blight/
     │   ├── Healthy/
     │   ├── Late_Blight/
     │   ├── Septoria_Leaf_Spot/
     │   └── Yellow_Leaf_Curl_Virus/
     └── Test/
         ├── Bacterial_Spot/
         ├── Early_Blight/
         ├── Healthy/
         ├── Late_Blight/
         ├── Septoria_Leaf_Spot/
         └── Yellow_Leaf_Curl_Virus/
     ```
   - Update the `data_dir` variable in `train.py` and `evaluation.py` to point to your dataset directory.

## Training the Model

1. Run the training script:
   ```
   python train.py
   ```
   This will perform k-fold cross-validation and save the best model for each fold.

2. The training process will output the following:
   - Training and validation accuracy for each epoch
   - Best validation accuracy for each fold
   - Average accuracy across all folds

## Evaluating the Model

1. After training, run the evaluation script:
   ```
   python evaluation.py
   ```

2. This will generate the following in the `plots` directory:
   - Confusion matrix
   - ROC curve

## Running the Web Application

1. Ensure you have the trained model file(s) in the project directory.

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and go to `http://localhost:5000` to use the web interface for making predictions.

## Project Structure

- `train.py`: Script for training the model using k-fold cross-validation
- `evaluation.py`: Script for evaluating the trained model
- `model.py`: Contains the model architecture definition
- `app.py`: Flask web application for serving predictions
- `remedies.json`: Contains information about disease symptoms, remedies, and prevention
- `templates/index.html`: HTML template for the web interface (not provided in the given files)

## Notes

- Make sure to adjust the `data_dir` path in `train.py` and `evaluation.py` to match your dataset location.
- The web application uses the `best_model_fold_0.pth` by default. Modify `app.py` if you want to use a different model or an ensemble.
- Ensure that your system has sufficient GPU resources for training, or adjust the batch size and image dimensions accordingly for CPU training.
