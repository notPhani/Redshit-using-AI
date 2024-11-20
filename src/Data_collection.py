# Importing the base modules
import pandas as pd #Used for reading the data from CSV
import numpy as np
import os
import shutil
import glob
import time
import bz2
from colorist import Color
import datetime

# Import the web automation library
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
#Folder monitoring 
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
# Intitializing the options to an empty function to prevent errors
options = Options()

# Enter the SDSS data collection webpage
 # and m = n
class Id_collection:
    def __init__(self,n_points,csv_file):
        self.n_points = n_points
        self.csv_file = csv_file
    def get_Ids(self):
        file = pd.read_csv(self.csv_file)
        ids = file["specobj_id"].head(self.n_points).tolist()
        SpaceIds = []
        for i in range(len(ids)):
            SpaceIds.append(int(ids[i][1:-1]))
        return SpaceIds
    def url_const_im(self,SpaceIds):
        urls = []
        for i in range(len(SpaceIds)):
            urls.append(f"http://cas.sdss.org/dr18/VisualTools/explore/summary?sId={SpaceIds[i]}")
        return urls
    def url_const_sp(self,SpaceIds):
        urls = []
        for i in range(len(SpaceIds)):
            urls.append(f"https://cas.sdss.org/dr18/VisualTools/explore/fitsspec?spec={SpaceIds[i]}")
        return urls
    def is_present(driver,url):
        element = driver.find_element(By.XPATH,'//*[@id="explore"]/div[3]/div/div/text()')
        if element:
            return False
        else:
            return True
class Unpack_move:
    def unpack_fits(file_path):
        # Create a unique filename based on the current timestamp
        timestamp = datetime.datetime.now().strftime("%M%S")
    
        # Check if the filename contains parentheses
        if '(' in os.path.basename(file_path) and ')' in os.path.basename(file_path):
            fits_file_n = f"{file_path[:-11]}_{timestamp}.fits"  # Adjust as necessary based on your naming convention
        else:
            fits_file_n = f"{file_path[:-8]}_{timestamp}.fits"  # Adjust as necessary based on your naming convention
    
    # Unpack the .bz2 file
        with bz2.open(file_path, "rb") as file:
            with open(fits_file_n, "wb") as fits_file:
                fits_file.write(file.read())
    
        return fits_file_n


    def move_rename_images(file_name, id):
        collection_unit = os.path.join("C:/Users/HP/Desktop/Python/Project/data", f"Galaxy_{id}/images")
        os.makedirs(collection_unit, exist_ok=True)
        shutil.move(file_name, os.path.join(collection_unit, os.path.basename(file_name)))

    def move_rename_spectrum(file_name, id):
        collection_unit = os.path.join("C:/Users/HP/Desktop/Python/Project/data", f"Galaxy_{id}/spectrum")
        os.makedirs(collection_unit, exist_ok=True)
        shutil.move(file_name, os.path.join(collection_unit, os.path.basename(file_name)))


# Initializing monitoring systems
class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            if os.path.basename(event.src_path).endswith(".fits.bz2"):
                print(f"{event.src_path} is {Color.GREEN}downloaded")

def monitor_folder(folder_path):
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"{Color.GREEN}Started monitoring folder: {Color.CYAN}{folder_path}")
    try:
        while True:
            time.sleep(1)  
    except KeyboardInterrupt:
        pass
        observer.stop()  # Stop the observer when interrupted (Ctrl+C)
        #print("Stopped monitoring.")
    observer.join()
    

    
# Intitialize the Id collection class
Id_collection = Id_collection(504,"C:/Users/HP/Desktop/Python/Project/main copy/optical_search_430636.csv")
spaceIds = Id_collection.get_Ids()
#print(spaceIds)
url_collection = Id_collection.url_const_im(spaceIds)
#print(url_collection)
url_collection_spectrum = Id_collection.url_const_sp(spaceIds)
#print(url_collection_spectrum)

# Now we have the urls for all n_points number of galxies in a list
# Next part is to access the url and download the filters and spectrum

def navigate_to_fits(url):
    driver = webdriver.Firefox()
    driver.get(url)

    # Wait until the "FITS" link becomes clickable
    wait = WebDriverWait(driver, 15)
    fits_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'FITS')))
    fits_link.click()

    # Switch to the new tab
    driver.switch_to.window(driver.window_handles[1])

    # Download all filters in corrected frames
    filters = ["u", "g", "r", "i", "z"]

    # Collect and click all download links without waiting
    for filter in filters:
        wait_time = WebDriverWait(driver, 3.5)
        download_link = wait_time.until(EC.presence_of_element_located((By.LINK_TEXT, filter)))
        download_link.click()
        time.sleep(0.01)  # Click the link directly to start download

    # Wait for the downloads to complete
    downloads_path = "C:/Users/HP/Downloads"

    # Check for the number of downloaded files
    while True:
        file_list = glob.glob(os.path.join(downloads_path, "*.bz2"))
        
        if len(file_list) >= 5:  # Check if at least 5 files have been downloaded
            print(f"5 files downloaded {Color.GREEN}Successfully")
            break

        time.sleep(1)
    time.sleep(0.5)

    driver.quit()  # Close the browser window

def navigate_to_spectrum(url):
    driver = webdriver.Firefox()
    driver.get(url)
    wait = WebDriverWait(driver,2.5)
    folder_path = "C:/Users/HP/Downloads"
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"{Color.GREEN}Started monitoring folder: {Color.CYAN}{folder_path}")

    spectrum_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT,"Download")))
    spectrum_link.click()
    time.sleep(1)
    driver.quit()
    observer.stop()
    print(f"Spectrum downloaded {Color.GREEN}Successfully")
    observer.join()  
#running a loop
downloads_path = "C:/Users/HP/Downloads"
for i in range(100):
    print(f"Started downloading Galaxy{i+1}_{spaceIds[i]}")
    navigate_to_fits(url_collection[i])
    file_list = glob.glob(os.path.join(downloads_path,"*.bz2"))
    file_list.sort(key = os.path.getmtime,reverse=True)
    file_list = file_list[0:5]
    folder_id = spaceIds[i]
    for j in range(len(file_list)):
        ftbu = file_list[j]
        unpacked_file = Unpack_move.unpack_fits(ftbu)
        #moving to folder
        Unpack_move.move_rename_images(unpacked_file,folder_id)
        print(f"{Color.GREEN}Moved {unpacked_file} into Galaxy{folder_id}/images ")
    navigate_to_spectrum(url_collection_spectrum[i])
    spec_file_list = glob.glob(os.path.join(downloads_path,"*fits"))
    spec_file_list.sort(key = os.path.getmtime,reverse=True)
    spec_file_list = spec_file_list[0]
    Unpack_move.move_rename_spectrum(spec_file_list,folder_id)
    print(f"{Color.GREEN}Moved {spec_file_list} into Galaxy{folder_id}/spectrum ")
