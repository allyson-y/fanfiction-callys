import csv
import os  
import fnmatch  
import pandas as pd
from pandas import *
import operator
import nltk
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def txtToCSV():
    # Creates and cleans the fanfiction txt files from fanfic-pack directory
    # Stores certain aspects of the fanfiction file and adds that information to a row in Fanfictions.csv
    # Removes rows with Nan genres or summaries
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
                    value = line[len(field[field_i] + ":"):len(line) - 1]
                    file_info.append(value)
                    print(f"file {filename} with field {field[field_i]} with line {line}")
                    if field_i < len(field) - 1:
                        field_i += 1
                    else:
                        break
            writer.writerow(file_info)
            file.close()

def removeNans(csv):
    print(len(csv))
    csv = csv.dropna()
    print(len(csv))
    return csv

def getGenres(csv):
    # Collects all the genres from the csv and compiles them into a dictionary of genre counts
    genresRanked = {}
    genresDict = {}
    genres = csv['Genre'].to_list()
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
    topGenres = genresRanked[:20]
    return topGenres.keys()

def classify(docs, cats):
    # Loading in features using a shortcut (CountVectorizer and TfidfVectorizer are both useful for quickly creating lexical features)
    cv = CountVectorizer(min_df=10, max_df=0.4) # words must show up in at least 5 and no more than 40% of documents
    features = cv.fit_transform(docs)
    print(features.shape)
    vocab = cv.get_feature_names_out()
    print(vocab)

    # Training a classifier
    # If you want to use another classifier here, just check out sklearn's options:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # You can also configure these by giving arguments - see the docs
    mnb = MultinomialNB()
    feat_train, feat_test, label_train, label_test = train_test_split(features, cats, test_size=0.2)
    mnb.fit(feat_train, label_train)
    preds = mnb.predict(feat_test)

    # Confusion matrix format (TP = true positive, FN = false negative, etc.)
    # [[TP, FP],
    #  [FN, TN]]
    print("Confusion Matrix)")
    print(confusion_matrix(label_test, preds))

    # Compute logs odd ratio: log p(pos|w) - log p(neg|w)
    log_odds = mnb.feature_log_prob_[1] - mnb.feature_log_prob_[0]
    wd_list = sorted([i for i in zip(log_odds, vocab)])

    print(wd_list[:50])

def main():
    # txtToCSV() # called already 69515
    fanfictions = pd.read_csv("Fanfictions.csv")
    cleanedFanFictions = removeNans(fanfictions)
    # print(len(fanfictions))
    genres = getGenres(cleanedFanFictions)
    print(cleanedFanFictions)
    classify(cleanedFanFictions["Summary"], cleanedFanFictions["Genre"])


if __name__ == "__main__":
    main()
