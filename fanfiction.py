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

from collections import Counter

import tomotopy
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns

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
    topGenres = genresRanked[:10]
    return topGenres

def getSummaries(csv, topGenres):
    topSummaries = []
    summaries = csv['Summary'].to_list()
    genresTog = csv['Genre'].to_list()
    tuples = [(key, value) for i, (key, value) in enumerate(zip(summaries, genresTog))]
    sum_genres = dict(tuples)
    print(sum_genres)
    for i in range(0, len(sum_genres)):
        for g in topGenres:
            if g in sum_genres.values()[i]:
                topSummaries.append(sum_genres.keys()[i])
    return topSummaries

def classify(summaries, genres):
    # Loading in features using a shortcut (CountVectorizer and TfidfVectorizer are both useful for quickly creating lexical features)
    cv = CountVectorizer(min_df=10, max_df=0.4) # words must show up in at least 5 and no more than 40% of documents
    features = cv.fit_transform(summaries)
    print(features.shape)
    vocab = cv.get_feature_names_out()
    print(vocab)

    # Training a classifier
    # If you want to use another classifier here, just check out sklearn's options:
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # You can also configure these by giving arguments - see the docs
    mnb = MultinomialNB()
    feat_train, feat_test, label_train, label_test = train_test_split(features, genres, test_size=0.2)
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

def clusterTopicVectors(fanfictions):
    # 10-topic model populated with documents from our reviews (with stopwords removed)
    n_topics = 10
    n_docs = len(fanfictions)

    # stoplist = set(stopwords.words('the'))
    mdl = tomotopy.LDAModel(k=n_topics)
    for summary in fanfictions["Summary"]:
        mdl.add_doc([summary])
    
    # Most converging will happen fast, but we'll run for 1000 iterations just in case
    # (this will take a minute)
    iters_per_check = 50
    for i in range(0, 1000, iters_per_check):
        mdl.train(iters_per_check)
        print('Iteration: {}\tLog-likelihood: {}'.format(i+iters_per_check, mdl.ll_per_word))

    # Print top 10 summaries of each topic
    print("Top 10 summaries by topic")
    for k in range(n_topics):
        print('#{}: {}'.format(k, ' '.join([w for (w, prop) in mdl.get_topic_words(k, top_n=10)])))
    
    return [mdl.get_topic_words(k, top_n=10) for k in range(n_topics)]

def genreProbabilities(genres, topic_words):
    # Step 2: Calculate probabilities for genre words in each topic
    genre_probabilities_per_topic = {}
    for genre in genres:
        genre_probabilities_per_topic[genre] = {}
        for topic_id, top_words in enumerate(topic_words):
            genre_probabilities_per_topic[genre][topic_id] = 0.0
            # print("TOP WORDS")
            # print(top_words[0])
            for word, probability in top_words:
                if genre in word:
                    genre_probabilities_per_topic[genre][topic_id] += probability
    
    print(genre_probabilities_per_topic)

    # Step 3: Normalize probabilities
    for genre, probabilities in genre_probabilities_per_topic.items():
        total_probability = sum(probabilities.values())
        for topic_id in probabilities:
            genre_probabilities_per_topic[genre][topic_id] /= total_probability

    # Step 4: Assign genres to topics
    topics_with_genres = {}
    for topic_id in range(num_topics):
        topics_with_genres[topic_id] = []
        for genre, probabilities in genre_probabilities_per_topic.items():
            if probabilities[topic_id] > threshold:  # Set a threshold for probability
                topics_with_genres[topic_id].append(genre)

    print(topics_with_genres)


def main():
    # txtToCSV() 
    fanfictions = pd.read_csv("Fanfictions.csv")
    cleanedFanFictions = removeNans(fanfictions)
    print(cleanedFanFictions)
    # print(len(fanfictions))
    genresCount = getGenres(cleanedFanFictions)
    genres = [genreCount[0] for genreCount in genresCount]
    print(genres)
    # summaries = getSummaries(cleanedFanFictions, genres)
    # print(summaries)
    topic_words = clusterTopicVectors(cleanedFanFictions)
    print("TOPIC WORDS [1]")
    print(topic_words[1])
    # classify(cleanedFanFictions["Summary"], cleanedFanFictions["Genre"])
    print(genreProbabilities(genres, topic_words))


if __name__ == "__main__":
    main()
