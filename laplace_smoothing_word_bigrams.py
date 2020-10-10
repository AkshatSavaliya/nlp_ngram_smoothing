import math
import re
from collections import Counter
import os

def get_bigrams(lang_file):
    language_file = open(lang_file, 'r')
    text = language_file.read().lower()
    language_file.close()
    text = '<s> ' + text
    text = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', text))))))
    words = text.split()
    words[-1] = words[-1][:-1]
    words.append("</s>")
    split_index = int(len(words)*0.999)
    word_dict = {i:True for i in list(set(words[0:split_index]))}
    training_words = words[0:split_index]
    held_out_words = [i if i in word_dict else "<UNK>" for i in words[split_index:]]
    training_words.extend(held_out_words) 
    word_counts = Counter(training_words)
    bigrams = [(training_words[i], training_words[i+1]) for i in range(len(training_words)-1)]
    bigram_counts = Counter(bigrams)
    return bigram_counts, word_counts

def get_probabilities(test_file, bigram_counts, word_counts):
    test_data = open(test_file, 'r')
    probability_list = []
    for m in test_data:
        probability = 0.0
        data_words = m.lower()
        data_words = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', data_words))))))
        test_words = [i if i in word_counts else "<UNK>" for i in data_words.split()]
        test_words.pop(0)
        test_words.pop(-1)
        test_bigrams = [(test_words[i], test_words[i+1]) for i in range(len(test_words)-1)]
        for j in test_bigrams:
            if j in bigram_counts:
                probability += math.log((bigram_counts[j] + 1) / (word_counts[j[0]] + len(word_counts)))
            else:
                probability += math.log(1 / (word_counts[j[0]] + len(word_counts)))
        probability_list.append((math.exp(probability)))
        probability = 0.0
    test_data.close()
    return probability_list

def predict_language(languages, probabilities):
    predictions = []
    row = 1
    guess = languages[0]
    for i in range(len(probabilities[0])):
        high = 0.0
        for j in range(len(probabilities)):
            if probabilities[j][i] > high: 
                high = probabilities[j][i]
                guess = languages[j]
        predictions.append((row, guess))
        row += 1
    predictions.insert(0, ('ID', 'LANG'))
    return predictions

def read_labels(label_file):
    labels = open(label_file, 'r')
    label_list = [''.join(k for k in i if k.isalpha()) for i in labels]
    labels.close()
    return label_list

def find_accuracy(pred, labels):
    denominator = len(pred) - 1
    numerator = 0
    for i in range(1, denominator):
        numerator += min(1, int(pred[i][1] == labels[i]))
    accuracy = numerator / denominator
    return (numerator, denominator, accuracy)

dir_path = os.path.dirname(os.path.realpath(__file__))
labels = read_labels(dir_path + '\\LangID.gold.txt')
language_names = ['EN', 'FR', 'GR']
en_bigrams, en_words = get_bigrams(dir_path + '\\EN.txt')
fr_bigrams, fr_words = get_bigrams(dir_path + '\\FR.txt')
gr_bigrams, gr_words = get_bigrams(dir_path + '\\GR.txt')
en_probs = get_probabilities(dir_path + '\\LangID.test.txt', en_bigrams, en_words)
fr_probs = get_probabilities(dir_path + '\\LangID.test.txt', fr_bigrams, fr_words)
gr_probs = get_probabilities(dir_path + '\\LangID.test.txt', gr_bigrams, gr_words)
predictions = predict_language(language_names, [en_probs, fr_probs, gr_probs])
accuracy = find_accuracy(predictions, labels)

print(predictions)
print("\nCorrect Tests: {}\nTotal Tests: {}\nAccuracy: {acc:%}".format(accuracy[0], accuracy[1], acc = accuracy[2]))