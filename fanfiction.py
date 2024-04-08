import csv
import os  
import fnmatch  
import pandas as pd
from pandas import *
import operator
import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
import string

from collections import Counter

import tomotopy
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets._samples_generator import make_blobs
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

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

def preprocess(input_file):
    #preprocessing: removing na, lowercasing, removing non english summaries
    def get_lang_detector(nlp, name):
        return LanguageDetector()

    nlp = spacy.load("en_core_web_sm")
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('language_detector', last=True)

    output_file = "cleanedFanFictions.csv"
    with open(input_file, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        with open(output_file, 'w', newline='') as csvfile_out:
            csvwriter = csv.writer(csvfile_out)
            for row in csvreader:
                if not any(pd.isnull(cell) for cell in row):
                    doc = nlp(row[-1])
                    detect_language = doc._.language
                    if detect_language['language'] == "en":
                        lowercase_row = [cell.lower() for cell in row]
                        print(lowercase_row)
                        csvwriter.writerow(lowercase_row)

def getGenres(csv):
    # Collects all the genres from the csv and compiles them into a dictionary of genre counts
    genresRanked = {}
    genresDict = {}
    genres = csv['genre'].to_list()
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
    topGenres = dict(genresRanked[:10])

    print("savefig")
    #Plotting
    fig = plt.figure(figsize =(10, 7))
    plt.bar(topGenres.keys(), topGenres.values())
    plt.xlabel('Genres')
    plt.ylabel('Counts')
    plt.title('Top Fan Fiction Genres by Count')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('genres.png')
    plt.close()

    return topGenres

def getSummaries(csv, topGenres):
    topSummaries = []
    summaries = csv['summary'].to_list()
    genresTog = csv['genre'].to_list()
    tuples = [(key, value) for i, (key, value) in enumerate(zip(summaries, genresTog))]
    sum_genres = dict(tuples)
    print(sum_genres)
    for i in range(0, len(sum_genres)):
        for g in topGenres:
            if g in sum_genres.values()[i]:
                topSummaries.append(sum_genres.keys()[i])
    return topSummaries

def clusterTopicVectors(fanfictions, n_topics=10):
    # topic model populated with documents from our reviews (with stopwords removed)
    n_docs = len(fanfictions)

    stoplist = set(stopwords.words('english'))
    mdl = tomotopy.LDAModel(k=n_topics)
    for summary in fanfictions["summary"]:
        if not isinstance(summary, float): #nan
            sentences = word_tokenize(summary)
            mdl.add_doc([word for word in sentences if word not in stoplist and word not in string.punctuation and '\'' not in word]) #using ntlk's word_tokenize
    
    # Most converging will happen fast, but we'll run for 1000 iterations just in case
    # (this will take a minute)
    iters_per_check = 50
    for i in range(0, 1000, iters_per_check):
        mdl.train(iters_per_check)
        print('Iteration: {}\tLog-likelihood: {}'.format(i+iters_per_check, mdl.ll_per_word))

    # Print top 10 words of each topic
    print(f"Top 10 words for {n_topics} topics")
    for k in range(n_topics):
        print('#{}: {}'.format(k, ' '.join([w for (w, prop) in mdl.get_topic_words(k, top_n=10)])))

    # Create an empty NumPy array to store the topic distributions
    topic_distributions = np.empty((n_docs, n_topics))

    # Fill the NumPy array with the topic distributions for each document
    for i, doc in enumerate(mdl.docs):
        topic_dist = doc.get_topic_dist()
        # print(topic_dist)
        topic_distributions[i] = [topic_dist[k] for k in range(n_topics)]
    
    return topic_distributions

def graphClusters(topic_dist):
    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(topic_dist)
    y_kmeans = kmeans.predict(topic_dist)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_topic_dist = pca.fit_transform(topic_dist)
    
    plt.scatter(reduced_topic_dist[:, 0], reduced_topic_dist[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = pca.transform(kmeans.cluster_centers_)  # Transform cluster centers to the reduced dimension space
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title("Cluster Plot of Topics")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    plt.savefig('topicClusters.png')
    plt.close()
    # kmeans = KMeans(n_clusters=15, random_state=0, n_init="auto").fit(topic_dist)
    # y_kmeans = kmeans.predict(topic_dist)
    # plt.scatter(topic_dist[:, 0], topic_dist[:, 1], topic_dist[:, 2], c=y_kmeans, s=50, cmap='viridis')
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.savefig('topicClusters.png')
    # plt.close()


def genreProbabilities(fanfictions, genres):
    n_docs = len(fanfictions)
    n_genres = len(genres)
    # Create an empty NumPy array to store the topic distributions
    genre_distributions = np.empty((n_docs, ), dtype=int)
    labels = [i for i in range(n_genres)]
    print(labels)

    for i in range(len(fanfictions["summary"])):
        if not isinstance(fanfictions["summary"][i], float): #nan
            for j in range(n_genres):
                if genres[j] in fanfictions["genre"][i]:
                    genre_distributions[i] = j
                    break

    return genre_distributions

def classify(topic_distributions, genre_distributions):
    gnb = GaussianNB()

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize an array to store the accuracies
    accuracies = []

    for train_index, test_index in kf.split(topic_distributions):
        X_train, X_test = topic_distributions[train_index], topic_distributions[test_index]
        y_train, y_test = genre_distributions[train_index], genre_distributions[test_index]

        # Train the classifier on the training set
        gnb.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = gnb.predict(X_test)

        # Calculate the accuracy of the classifier
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Calculate the average accuracy over the 5 folds
    average_accuracy = sum(accuracies) / len(accuracies)
    print("Average accuracy:", average_accuracy)

def main():
    # txtToCSV() #preprocessing
    # cleanedFanFictions = preprocess("Fanfictions.csv") #preprocessing
    # print(cleanedFanFictions)
    fanfictions = pd.read_csv("Fanfictions.csv")
    print(len(fanfictions))
    cleanedFanFictions = pd.read_csv("cleanedFanFictions.csv")
    print(len(cleanedFanFictions))
    genresCount = getGenres(cleanedFanFictions)
    print(genresCount)
    genres = list(genresCount.keys())
    topic_distributions = clusterTopicVectors(cleanedFanFictions)
    print("TOPIC DISTRIBUTIONS") # num fan fiction rows x num topic columns
    print(topic_distributions)
    genre_distributions = genreProbabilities(cleanedFanFictions, genres)
    print("GENRE DISTRIBUTIONS") # num fan fiction rows x num genres
    print(genre_distributions)
    classify(topic_distributions, genre_distributions)
    graphClusters(topic_distributions)


if __name__ == "__main__":
    main()
