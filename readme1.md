# Galaxy Redshift Prediction Using CNN

## Project Overview

This project leverages **Convolutional Neural Networks (CNNs)** to predict the **redshift** of galaxies from their spectra and filter images. By training the model on galaxy filter data and corresponding spectra, we aim to reconstruct the galaxy spectrum and use it to predict the redshift, an important astronomical parameter that helps us understand the distance and velocity of galaxies.

### Key Features:
- **End-to-End Workflow**: From data preprocessing to model training and redshift prediction, this project implements a complete machine learning pipeline.
- **CNN-Based Spectrum Reconstruction**: The model uses CNN layers to reconstruct galaxy spectra from multi-filter images.
- **Redshift Prediction**: After reconstructing the spectrum, we predict the redshift, which is a key aspect of studying galaxies.
- **Data Augmentation**: To improve model robustness and prevent overfitting, data augmentation techniques are applied during training.
- **Comprehensive Model Evaluation**: Includes detailed loss metrics, MAE (Mean Absolute Error), and graphical visualizations comparing predicted vs. true spectra.
- **Automated Data Handling**: Scripts for dynamically downloading, cleaning, and processing large datasets of galaxy images and spectra.
- **Model Performance**: The model has been trained for 2000 epochs, showing strong performance in redshift prediction based on reconstructed spectra.

## Getting Started

To get started with the project, follow the instructions below to set up the environment and run the model.

### Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.x
- pip (Python package manager)

### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/galaxy-redshift-prediction.git
cd galaxy-redshift-prediction
(Optional but recommended) Create a virtual environment:
bash
Copy code
python -m venv venv
Install the project dependencies:
bash
Copy code
pip install -r requirements.txt
How to Use
Prepare the data: Ensure that your galaxy filter image files are organized in the correct folder structure as expected by the code (refer to script comments or documentation for guidance).

Train the model: Start training the model using the main.py script. This script handles data preprocessing and model training.

bash
Copy code
python main.py
Make Predictions: Once the model is trained, you can use the main_spec.py and main_redshift.py scripts to predict galaxy spectra and redshift, respectively.
bash
Copy code
python main_spec.py    # Predict the reconstructed spectrum of galaxies
python main_redshift.py  # Predict the redshift based on the reconstructed spectrum
Evaluate Results: The scripts will visualize the predicted spectra alongside the true spectra, and output metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).
Directory Structure
bash
Copy code
/galaxy-redshift-prediction
│
├── /data                    # Folder containing galaxy filter and spectrum data.
├── /models                  # Folder where trained models are saved.
├── /notebooks               # Jupyter notebooks for analysis and experimentation.
├── /src                     # Source code for data processing, training, and prediction.
│
├── main.py                 # Main script to train the model.
├── main_spec.py            # Script to predict the reconstructed spectrum of galaxies.
├── main_redshift.py        # Script to predict the redshift based on the reconstructed spectrum.
├── requirements.txt        # Python dependencies for the project.
├── LICENSE                 # License file for the project (MIT or similar).
└── README.md               # This file.
Model Details
The model is a Convolutional Neural Network (CNN) designed to process multi-filter galaxy images and reconstruct their spectra. This reconstructed spectrum is then used to predict the galaxy's redshift, a key parameter in understanding the galaxy's movement and distance in the universe.

Convolutional Layers: The CNN uses convolutional layers to extract features from the filter images.
Fully Connected Layers: After feature extraction, fully connected layers are used to predict the redshift.
Optimizer: Adam optimizer is used with a learning rate scheduler to improve training performance.
Results
The model performs well on the training data, showing a low error rate in spectrum reconstruction and accurate redshift predictions. The results are evaluated based on Mean Squared Error (MSE) and Mean Absolute Error (MAE), providing an overall assessment of prediction accuracy.

Contributing
We welcome contributions! If you’d like to improve this project or add new features, follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Push your changes and create a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
MLH Fellowship: This project is part of my submission for the MLH Fellowship program.
Kaggle: Some datasets and resources were sourced from Kaggle for this project.
TensorFlow/Keras: For the deep learning framework used to build and train the model.
Contact
Feel free to reach out for any inquiries or collaboration opportunities:

GitHub: [Your GitHub Profile Link]
Email: [Your Email Address]
