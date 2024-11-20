import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.models import load_model
import numpy as np
from scipy.interpolate import interp1d
from utils import extract_Ra_Dec as Extract
from utils import isolate_galaxy as isolate
from utils import find_pixels as find
from utils import softening_image as smooth
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


model = load_model("C:/Users/HP/Desktop/Python/Project/final_model.keras")

def resample(wave, flux, target_length=4000, epsilon=1e-6):
    # Ensure that the minimum wavelength is greater than zero before applying log10
    wave_min = wave.min()
    wave_max = wave.max()
    
    # If there are non-positive values in wave, adjust them
    if wave_min <= 0:
        wave_min = np.max([wave_max * epsilon, epsilon])  # Make sure it's positive
    if wave_max <= 0:
        wave_max = np.max([wave_min * (1 / epsilon), 1 / epsilon])  # Avoid zero or negative

    # Apply log10 with adjusted minimum and maximum values
    new_waves = np.logspace(np.log10(wave_min), np.log10(wave_max), target_length)
    
    # Interpolate the flux based on the new wavelengths
    interp_func = interp1d(wave, flux, kind="linear", fill_value="extrapolate")
    new_flux = interp_func(new_waves)
    
    return new_waves, new_flux



def stack_input(folder_name):
    spec_folder = os.path.join(folder_name,"spectrum")
    filter_folder = os.path.join(folder_name,"images")

    filters = os.listdir(filter_folder)
    spectrum = os.listdir(spec_folder)[0]

    ra, dec = Extract(os.path.join(spec_folder, spectrum))
    x_pixel, y_pixel = find(os.path.join(filter_folder, filters[0]), ra, dec)

    softened_images = []

    for i in range(len(filters)):
        file_path = os.path.join(filter_folder, filters[i])
        isolated_image = isolate(file_path, x_pixel, y_pixel, show=False)
        softened_image = smooth(isolated_image, show=False)
        softened_images.append(softened_image)
    
    try:
        shapes = [img.shape for img in softened_images]
        if len(set(shapes)) > 1:
            print(f"Your images are in different shapes")
        else:
            stacked_images = np.stack(softened_images, axis=0)  # Fix this line
            return stacked_images
    except Exception as e:
        print(f"Error stacking images: {e}")



def get_spectrum(file_path,show=False):

    with fits.open(file_path,mode="readonly") as file:
        # Assuming the spectrum is in the second HDU
        data = file[1].data["model"]
        waves = 10**file[1].data["loglam"]
        normalized_model = np.log10(data+1e-10)
        #print(normalized_model.shape)# 'flux' column contains the spectrum data
        #flux_length = len(data)
    if show == False:
        return normalized_model,waves
    else:
        plt.plot(waves,normalized_model)
        plt.show()
        return normalized_model,waves



def test_data(galaxy_folder):
    spectrum_folder = f"{galaxy_folder}/spectrum"
    spec_file = os.path.join(spectrum_folder,os.listdir(spectrum_folder)[0])

    x_test = stack_input(galaxy_folder)
    _,data = get_spectrum(spec_file)
    y_test = resample(_,data,4000)
    y_test = y_test/np.max(y_test)
    x_test = np.expand_dims(x_test,axis=-1)

    return x_test,y_test


def prep_data(galaxy_folder):
    galaxy_id = galaxy_folder[-len("299527610596091904"):]
    spec_folder = os.path.join(galaxy_folder,"spectrum")
    spec_file = os.path.join(spec_folder,os.listdir(spec_folder)[0])
    waves,x_spec = get_spectrum(spec_file)
    _,x_train = resample(waves,x_spec,4000)
    x_train = x_train/np.max(x_train)

    # x_train here is the spectrum of the galaxy and y_train is the reshift
    with fits.open(spec_file) as file:
        data = file[2].data  
        redshift = data["Z"]
        y_train = redshift
    
    return x_train,y_train


x_train = []
y_train = []

galaxy_folders = os.listdir("C:/Users/HP/Desktop/Python/Project/data")
for folder in galaxy_folders:
    folder = os.path.join("C:/Users/HP/Desktop/Python/Project/data",folder)
    x_temp,y_temp = prep_data(folder)
    x_train.append(x_temp)
    y_train.append(y_temp)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = x_train[-100:-1]
y_test = y_train[-100:-1]
x_train = x_train[:]
y_train = y_train[:]

dtrain = xgb.DMatrix(x_train,label=y_train)
dtest = xgb.DMatrix(x_test,label=y_test)

params = {
    'objective': 'reg:squarederror',  # Regression task (predicting continuous value)
    'eval_metric': 'rmse',            # Root Mean Squared Error (RMSE) as the evaluation metric
    'max_depth': 6,                   # Maximum depth of the trees
    'eta': 0.1,                       # Learning rate
    'subsample': 0.8,                 # Fraction of data to use for each boosting round
    'colsample_bytree': 0.8           # Fraction of features to use for each boosting round
}

# Train the model
num_round = 5000  # Number of boosting rounds
early_stopping_rounds = 500  # Stop if no improvement after these many rounds
evals = [(dtrain, 'train'), (dtest, 'eval')]

model = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=early_stopping_rounds,verbose_eval=True)

# Save the model for later use
model.save_model('xgboost_model.json')

# y_pred is the predictions made by your model
# y_test is the actual redshift values from your test set
y_pred = model.predict(dtest)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, y_pred, c='blue', alpha=0.6)
plt.xlabel("Actual Redshift")
plt.ylabel("Predicted Redshift")
plt.title("Actual vs Predicted Redshift")
plt.show()


