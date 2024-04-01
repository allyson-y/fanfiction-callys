import csv
import os  
import fnmatch  
import pandas as pd

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

def main():
    # creates and fills the Fanfictions.csv file
    # txtToCSV() -- called already


if __name__ == "__main__":
    main()
