import math
import re
from collections import Counter
import os

def train_language_letters(lang_file):
    language_file = open(lang_file, 'r')
    text = language_file.read().lower()
    language_file.close()
    text = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', text))))))
    letters = list(text)
    letter_counts = Counter(letters)
    bigrams = [(letters[i], letters[i+1]) for i in range(len(letters)-1)]
    num_bigrams = len(bigrams)
    bigram_counts = Counter(bigrams)
    bigram_dict = dict(bigram_counts)
    bigram_dict["<UNK>"] = 0
    for i in list(bigram_counts):
        if bigram_dict[i] == 1:
            bigram_dict["<UNK>"] += 1
            del bigram_dict[i]
    bigram_probabilities = {i: (bigram_dict[i] / letter_counts[i[0]]) if i != "<UNK>" else bigram_dict[i] / num_bigrams for i in bigram_dict.keys()}
    return bigram_probabilities

def get_language_vocab(lang_file):
    language_file = open(lang_file, 'r')
    text = language_file.read().lower()
    language_file.close()
    return set(text)

def predict_language_letters(lang_names, lang_dicts, lang_vocs, test_file):
    test_data = open(test_file, 'r')
    predictions = []
    row = 1
    for m in test_data:
        mle = 0.0
        probability = 0.0
        prediction = lang_names[0]
        test_letters = m.lower()
        test_letters = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', test_letters))))))
        test_bigrams = [(test_letters[i], test_letters[i+1]) for i in range(len(test_letters)-1)]
        for i in range(len(lang_names)):
            for j in test_bigrams:
                probability = probability + math.log(lang_dicts[i][j]) if j in lang_dicts[i] else probability + math.log(lang_dicts[i]["<UNK>"])
            if math.exp(probability) > mle:
                mle = math.exp(probability)
                prediction = lang_names[i]
            probability = 0.0
        predictions.append((row, prediction))
        row += 1
    test_data.close()
    predictions.insert(0, ('ID', 'LANG'))
    return predictions

def read_labels(label_file):
    labels = open(label_file, 'r')
    label_list = [''.join(k for k in i if k.isalpha()) for i in labels]
    labels.close()
    return label_list

def find_accuracy(predictions, labels):
    denominator = len(labels) - 1
    numerator = 0
    for i in range(1, denominator):
        numerator += min(1, int(predictions[i][1] == labels[i]))
    accuracy = numerator / denominator
    return (numerator, denominator, accuracy)

dir_path = os.path.dirname(os.path.realpath(__file__))
language_names = ['EN', 'FR', 'GR']
language_files = [dir_path + '\\EN.txt', dir_path + '\\FR.txt', dir_path + '\\GR.txt']
language_dictionaries = [train_language_letters(i) for i in language_files]
language_vocs = [get_language_vocab(i) for i in language_files]
labels = read_labels(dir_path + '\\LangID.gold.txt')
language_predictions = predict_language_letters(language_names, language_dictionaries, language_vocs, dir_path + '\\LangID.test.txt')
accuracy = find_accuracy(language_predictions, labels)

print(language_predictions)
print("\nCorrect Tests: {}\nTotal Tests: {}\nAccuracy: {acc:%}".format(accuracy[0], accuracy[1], acc = accuracy[2]))