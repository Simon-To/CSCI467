from collections import Counter, defaultdict


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

def read_data(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            label, book, passage = line.strip().split('\t')
            dataset.append((passage.split(' '), label))
    return dataset





if __name__ == '__main__':
    train_data = read_data('train.tsv')
    dev_data = read_data('dev.tsv')
    test_data = read_data('test.tsv')
    newbooks_data = read_data('newbooks.tsv')
    returned = get_word_counts(train_data)
