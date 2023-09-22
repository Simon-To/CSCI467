"""Code for HW1 Problem 4: for Author Attribution with Naive Bayes."""
import argparse
import math
from collections import Counter, defaultdict
import sys
from tqdm import tqdm

import numpy as np

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-set', '-e', choices=['dev', 'test', 'newbooks'])
    parser.add_argument('--analyze-counts', '-a', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_data(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            label, book, passage = line.strip().split('\t')
            dataset.append((passage.split(' '), label))
    return dataset

def get_vocabulary(dataset):
    return list(set(word for (words, label) in dataset for word in words))

def get_label_counts(train_data):
    """Count the number of examples with each label in the dataset.

    We will use a Counter object from the python collections library.
    A Counter is essentially a dictionary with a "default value" of 0
    for any key that hasn't been inserted into the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object mapping each label to a count.
    """
    label_counts = Counter()
    ### BEGIN_SOLUTION 4a

    # For every training data (words: list, label: str)
    for data in train_data:
        # Add the number of words to the author associated with those words.
        label_counts[data[1]] += len(data[0])

    ### END_SOLUTION 4a
    return label_counts

def get_word_counts(train_data):
    """Count occurrences of every word with every label in the dataset.

    We will create a separate Counter object for each label.
    To do this easily, we create a defaultdict(Counter),
    which is a dictionary that will create a new Counter object whenever
    we query it with a key that isn't in the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object where keys are tuples of (label, word), mapped to
        the number of times that word appears in an example with that label
    """
    word_counts = defaultdict(Counter)
    ### BEGIN_SOLUTION 4a
    for data in train_data:
        # Returns a dictionary of where: key = word, value = counts
        word_counter = Counter(data[0])
        word_counts[data[1]].update(Counter(data[0]))
        # for word in word_counter:
        #     if word_counts.get((data[1], word)) == None:
        #         word_counts[(data[1], word)] = 1
    ### END_SOLUTION 4a
    return word_counts

def predict(words, label_counts, word_counts, vocabulary):
    """Return the most likely label given the label_counts and word_counts.

    Args:
        words: List of words for the current input.
        label_counts: Counts for each label in the training data
        word_counts: Counts for each (label, word) pair in the training data
        vocabulary: List of all words in the vocabulary
    Returns:
        The most likely label for the given input words.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    ### BEGIN_SOLUTION 4a
    lamda = 1

    # numpy-ise the label counts:
    label_counts_numpy = np.array(list(label_counts.values()))

    # Calculate log of pi:
    log_pi = np.log(label_counts_numpy) - np.log(np.sum(label_counts_numpy))


    # Calculate each label's word count in vocabulary
    label_word_count = defaultdict()
    for label in labels:
        label_specific = []
        for word in vocabulary:
            label_specific.append(word_counts[label][word])
        label_word_count[label] = np.sum(np.array(label_specific))

    # Calculate log of tao:



    log_tao = []
    for label in labels:
        # 1. Numerator:
        word_label_count = []
        for word in words:
            word_label_count.append(word_counts[label][word])
        numerator = np.log(np.array(word_label_count) + lamda)

        # 2. Denominator:
        denominator = label_word_count[label] + len(vocabulary)
        denominator = np.log(denominator)

        # 3. tao_x_y:
        log_tao_x_y = numerator - denominator
        log_tao_x_y = np.sum(log_tao_x_y)
        log_tao.append(log_tao_x_y)

    predicts = log_tao + log_pi
    prediction = labels[np.argmax(predicts)]
    return prediction







    # raise NotImplementedError
    ### END_SOLUTION 4a

def evaluate(label_counts, word_counts, vocabulary, dataset, name, print_confusion_matrix=False):
    num_correct = 0
    confusion_counts = Counter()
    for words, label in tqdm(dataset, desc=f'Evaluating on {name}'):
        pred_label = predict(words, label_counts, word_counts, vocabulary)
        confusion_counts[(label, pred_label)] += 1
        if pred_label == label:
            num_correct += 1
    accuracy = 100 * num_correct / len(dataset)
    print(f'{name} accuracy: {num_correct}/{len(dataset)} = {accuracy:.3f}%')
    if print_confusion_matrix:
        print(''.join(['actual\\predicted'] + [label.rjust(12) for label in label_counts]))
        for true_label in label_counts:
            print(''.join([true_label.rjust(16)] + [
                    str(confusion_counts[true_label, pred_label]).rjust(12)
                    for pred_label in label_counts]))


def get_label_counts_on_words(labels, word_counts, vocabulary):
    counts = defaultdict(int)
    for label in labels:
        counts[label] = np.sum(np.array([word_counts[label][word] for word in vocabulary]))
    return counts



def analyze_counts(label_counts, word_counts, vocabulary):
    """Analyze the word counts to identify the most predictive features.

    For each label, print out the top ten words that are most indictaive of the label.
    There are multiple valid ways to define what "most indicative" means.
    Our definition is that if you treat the single word as the input x,
    find the words with largest p(y=label | x).

    The print steps are provided. You only have to compute the defaultdict "p_label_given_word".
    The key of the defaultdict is the label, and the value of the defaultdict is a list of
    probabilities p(y=label | x) of each single word x.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    p_label_given_word = defaultdict(list)
    ### BEGIN_SOLUTION 4b
    lamda = 0


    label_word_count = []
    label_count = []
    for label in labels:
        label_specific = []
        for word in vocabulary:
            label_specific.append(word_counts[label][word])
        label_word_count.append(np.sum(np.array(label_specific)))
        label_count.append(label_counts[label])
    label_word_count = np.array(label_word_count)
    label_count = np.array(label_count)



    w_y = []
    for label in labels:
        label_row = []
        for word in vocabulary:
            label_row.append(word_counts[label][word])
        label_row = np.array(label_row) + lamda * 1
        label_row = label_row[np.newaxis, :]
        if len(w_y) == 0:
            w_y = label_row
        else:
            w_y = np.concatenate((w_y, label_row), axis=0)

    prob_w_g_y = np.transpose(w_y) / (label_word_count + lamda * len(vocabulary))
    prob_y = label_count / np.sum(label_count)
    numerator = np.transpose(prob_w_g_y * prob_y)
    prob_y_g_w = numerator / np.sum(numerator, axis=0)

    for i, label in enumerate(labels):
        prob = prob_y_g_w[i]
        p_label_given_word[label] = list(zip(vocabulary, prob))





    ### END_SOLUTION 4b

    for label in labels:
        print(f'Label {label}')
        sorted_scores = sorted(p_label_given_word[label],
                               key=lambda x: x[1], reverse=True)
        for word, p in sorted_scores[:10]:
            print(f'    {word}: {p:.3f}')


def main():
    train_data = read_data('train.tsv')
    dev_data = read_data('dev.tsv')
    test_data = read_data('test.tsv')
    newbooks_data = read_data('newbooks.tsv')

    vocabulary = get_vocabulary(train_data)  # The set of words present in the training data
    label_counts = get_label_counts(train_data)
    word_counts = get_word_counts(train_data)
    if OPTS.analyze_counts:
        analyze_counts(label_counts, word_counts, vocabulary)
    evaluate(label_counts, word_counts, vocabulary, train_data, 'train')
    if OPTS.evaluation_set == 'dev':
        evaluate(label_counts, word_counts, vocabulary, dev_data, 'dev', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'test':
        evaluate(label_counts, word_counts, vocabulary, test_data, 'test', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'newbooks':
        evaluate(label_counts, word_counts, vocabulary, newbooks_data, 'newbooks', print_confusion_matrix=True)

if __name__ == '__main__':
    OPTS = parse_args()
    main()

