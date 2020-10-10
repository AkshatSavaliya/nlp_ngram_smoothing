import math
import re
from collections import Counter
import os

def get_grams(lang_file):
    language_file = open(lang_file, 'r')
    text = language_file.read().lower()
    language_file.close()
    text = '<s> ' + text
    text = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', text))))))
    words = text.split()
    words[-1] = words[-1][:-1]
    words.append("</s>")
    split_index = int(len(words)*0.9999)
    word_dict = {i:True for i in list(set(words[0:split_index]))}
    training_words = words[0:split_index]
    held_out_words = [i if i in word_dict else "<UNK>" for i in words[split_index:]]
    training_words.extend(held_out_words) 
    word_counts = Counter(training_words)
    bigrams = [(training_words[i], training_words[i+1]) for i in range(len(training_words)-1)]
    bigram_counts = Counter(bigrams)
    trigrams = [(training_words[i], training_words[i+1], training_words[i+2]) for i in range(len(training_words)-2)]
    trigram_counts = Counter(trigrams)
    return trigram_counts, bigram_counts, word_counts

def get_probabilities(test_file, trigram_counts, bigram_counts, word_counts):
    test_data = open(test_file, 'r')
    probability_list = []
    for m in test_data:
        probability = 0.0
        data_words = m.lower()
        data_words = re.sub(r'  *', ' ', re.sub(r'\n', ' ', re.sub(r'[!?]|(\. )|(\.\n)', ' </s> <s> ', re.sub(r' \' ', ' ', re.sub(r'\'', ' \'', re.sub(r'[,;:\d\"]', ' ', data_words))))))
        test_words = [i if i in word_counts else "<UNK>" for i in data_words.split()]

        for i in range(2, len(test_words)):
            discount = 0.0
            n_gram_count = 0
            string_count = 0
            final_words = 0
            preceding_strings = 0
            total_n_grams = 0

            if (test_words[i-2], test_words[i-1], test_words[i]) in trigram_counts:
                n_gram_count = trigram_counts[(test_words[i-2], test_words[i-1], test_words[i])]
                string_count = bigram_counts[(test_words[i-2], test_words[i-1])]
                for j in trigram_counts:
                    final_words = final_words + 1 if (test_words[i-2] == j[0] and test_words[i-1] == j[1]) else final_words
                    preceding_strings = preceding_strings + 1 if test_words[i] == j[2] else preceding_strings
                total_n_grams = len(trigram_counts)

            elif (test_words[i-1], test_words[i]) in bigram_counts:
                discount = 0.75
                n_gram_count = bigram_counts[(test_words[i-1], test_words[i])]
                string_count = word_counts[test_words[i-1]]
                for j in bigram_counts:
                    final_words = final_words + 1 if test_words[i-1] == j[0] else final_words
                    preceding_strings = preceding_strings + 1 if test_words[i] == j[1] else preceding_strings
                total_n_grams = len(bigram_counts)

            else:
                discount = 0.75
                n_gram_count = word_counts[test_words[i]]
                string_count = len(word_counts)
                final_words = len(word_counts)
                preceding_strings = 1
                total_n_grams = len(word_counts)

            p = (max((n_gram_count - discount), 0)) / string_count
            _lambda = (discount / string_count) * final_words
            p_c = preceding_strings / total_n_grams

            probability += math.log(p + (_lambda * p_c))

        probability_list.append((math.exp(probability)))
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
en_trigrams, en_bigrams, en_words = get_grams(dir_path + '\\EN.txt')
fr_trigrams, fr_bigrams, fr_words = get_grams(dir_path + '\\FR.txt')
gr_trigrams, gr_bigrams, gr_words = get_grams(dir_path + '\\GR.txt')
en_probs = get_probabilities(dir_path + '\\LangID.test.txt', en_trigrams, en_bigrams, en_words)
fr_probs = get_probabilities(dir_path + '\\LangID.test.txt', fr_trigrams, fr_bigrams, fr_words)
gr_probs = get_probabilities(dir_path + '\\LangID.test.txt', gr_trigrams, gr_bigrams, gr_words)
predictions = predict_language(language_names, [en_probs, fr_probs, gr_probs])
accuracy = find_accuracy(predictions, labels)

print(predictions)
print("\nCorrect Tests: {}\nTotal Tests: {}\nAccuracy: {acc:%}".format(accuracy[0], accuracy[1], acc = accuracy[2]))