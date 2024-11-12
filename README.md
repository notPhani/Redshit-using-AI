# Galaxy Redshift Prediction Project

## 🛠️ Project Overview
This project predicts the redshift of galaxies by analyzing filter data and reconstructing galaxy spectra. It integrates advanced data preprocessing, convolutional neural networks, and machine learning models.

## 🗺️ Workflow
1. **Data Collection**:
   - Automated downloads of FITS files using Selenium and dynamic XPath locators.
   - Timestamp-based file management to handle duplicates.

2. **Preprocessing**:
   - Cropped galaxy images using WCS coordinate transformation.
   - Cleaned noisy data with `galmask`.

3. **Spectrum Reconstruction**:
   - Built a CNN to reconstruct the galaxy spectra from filter wavelengths.
   - Normalized filter data using logarithmic scaling.

4. **Redshift Prediction**:
   - Fed reconstructed spectra into an XGBoost model to predict redshift values.

## 📊 Results
- Achieved a spectral reconstruction accuracy of [add metric].
- Predicted redshift with an RMSE of [add metric].

## 📂 Repository Structure
- `/data`: Raw and preprocessed datasets.
- `/scripts`: Python scripts for data preprocessing, CNN training, and redshift prediction.
- `/notebooks`: Jupyter notebooks for visualization and experimentation.

## 🔧 Tools and Technologies
- Python, TensorFlow, XGBoost, Pandas, Astropy, Matplotlib, Selenium.
