# Importing the base modules
import pandas as pd #Used for reading the data from CSV
import numpy as np
import os
import shutil
import glob
import time
import bz2
from colorist import Color

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

class Unpack_move:
    def unpack_fits(file_path):
        fits_file_n = file_path[:-4]
        with bz2.open(file_path,"rb") as file:
            with open(fits_file_n,"wb") as fits_file:
                fits_file.write(file.read())
        return fits_file_n
    def move_rename_images(file_name,id):
        collection_unit = os.path.join("C:/Users/HP/Desktop/Python/data",f"Galaxy_{id}/images")
        os.makedirs(collection_unit,exist_ok=True)
        shutil.move(file_name,os.path.join(collection_unit),os.path.basename(file_name))
    def move_rename_spectrum(file_name,id):
        collection_unit = os.path.join("C:/Users/HP/Desktop/Python/data",f"Galaxy_{id}/spectrum")
        os.makedirs(collection_unit,exist_ok=True)
        shutil.move(file_name,os.path.join(collection_unit),os.path.basename(file_name))

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
Id_collection = Id_collection(100,"./optical_search_412544.csv")
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
    # Define an explicit wait time (e.g., 15 seconds)
    wait = WebDriverWait(driver, 15)

    # Wait until the "FITS" link becomes clickable
    fits_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, 'FITS')))

    # Click the "FITS" link once it's clickable
    fits_link.click()
    
    # Switchcing the tab
    driver.switch_to.window(driver.window_handles[1])

    # Download all filters in corrected frames
    filters = ["u","g","r","i","z"]
    folder_path = "C:/Users/HP/Downloads"
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"{Color.GREEN}Started monitoring folder: {Color.CYAN}{folder_path}")

    # Loop for all the filters and download the first link.
    for filter in filters:
        wait_time = WebDriverWait(driver,10)
        download_link = wait_time.until(EC.presence_of_element_located((By.LINK_TEXT,filter)))
        time.sleep(3)
        download_link.click()     
    time.sleep(4)
    driver.quit()
    observer.stop()
    print(f"Filters downloaded {Color.GREEN}Successfully")
    observer.join()

def navigate_to_spectrum(url):
    driver = webdriver.Firefox()
    driver.get(url)
    wait = WebDriverWait(driver,15)
    folder_path = "C:/Users/HP/Downloads"
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"{Color.GREEN}Started monitoring folder: {Color.CYAN}{folder_path}")

    spectrum_link = wait.until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT,"Download")))
    spectrum_link.click()
    time.sleep(2)
    driver.quit()
    observer.stop()
    print(f"Spectrum downloaded {Color.GREEN}Successfully")
    observer.join()  
#running a loop

for i in range(1):
    navigate_to_fits(url_collection[i])
    navigate_to_spectrum(url_collection_spectrum[i])


downloads_path = "C:/Users/HP/Downloads"

file_list = glob.glob(os.path.join(downloads_path,"*.fits.bz2"))
file_list.sort(key = os.path.getmtime,reverse = True)


#Make batches of 5
for i in range(len(file_list)):
    folder_id = i//5 + 1
    ftbu = file_list[i]
    unpacked_file = Unpack_move.unpack_fits(ftbu)
    #moving to folder
    Unpack_move.move_rename_images(unpacked_file,folder_id)
    print(f"{Color.GREEN}Moved {unpacked_file} into Galaxy{folder_id}/images ")

#Formating the spectrum into the respective files

spec_file_list = glob.glob(os.path.join(downloads_path,"*fits"))
spec_file_list.sort(key = os.path.getmtime,reverse=True)

for i in range(len(spec_file_list)):
    folder_id = i+1
    Unpack_move.move_rename_spectrum(spec_file_list[i],folder_id)
    print(f"{Color.GREEN}Moved {spec_file_list[i]} into Galaxy{folder_id}/spectrum ")
