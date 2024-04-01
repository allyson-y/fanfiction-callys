import csv
import os  
import fnmatch  
import pandas as pd

def createCSV():
    with open('Fanfictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["category", "genre", "language", "status", "published", "updated", "packaged", "rating", "chapters", "words", "publisher", "story URL", "author URL", "Summary"]
        writer.writerow(field)

def addFiles():
    files_dir = r'fanfic-pack'  
    files = os.listdir(files_dir) 

createCSV()
addFiles()