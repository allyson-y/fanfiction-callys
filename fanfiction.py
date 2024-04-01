import csv
import os  
import fnmatch  
import pandas as pd
from pandas import *
import operator

def txtToCSV():
    # Creates and cleans the fanfiction txt files from fanfic-pack directory
    # Stores certain aspects of the fanfiction file and adds that information to a row in Fanfictions.csv
    field = []
    with open('Fanfictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["File Name", "Category", "Genre", "Language", "Status", "Published", "Updated", "Packaged", "Rating", "Chapters", "Words", "Publisher", "Story URL", "Author URL", "Summary"]
        writer.writerow(field)
        files_dir = r'fanfic-pack'  
        # iterates through directory of csv files 
        for filename in os.listdir(files_dir):
            file = open(os.path.join(files_dir, filename), 'r')
            field_i = 1
            file_info = [filename]
            for line in file.readlines():
                if field[field_i] + ":" in line:
                    file_info.append(line[len(field[field_i] + ":"):len(line) - 1])
                    print(f"file {filename} with field {field[field_i]} with line {line}")
                    if field_i < len(field) - 1:
                        field_i += 1
                    else:
                        break
            writer.writerow(file_info)
            file.close()

def getGenres(filename):
    genresRanked = {}
    genresDict = {}
    fanfictions = read_csv(filename)
    genres = fanfictions['Genre'].to_list()
    # print(genres)
    for g in genres:
        g = str(g)
        words = g.split(', ')
        for w in words:
            w = w.strip()
            if w in genresDict.keys():
                genresDict[w] += 1
            else:
                genresDict[w] = 1 
    genresRanked = sorted(genresDict.items(), key= operator.itemgetter(1), reverse = True)
    return genresRanked[:20]




def main():
    # creates and fills the Fanfictions.csv file
    # txtToCSV() # called already
    print(getGenres("Fanfictions.csv"))


if __name__ == "__main__":
    main()
