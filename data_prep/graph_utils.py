from collections import defaultdict
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def word_windows(corpus, window_size=15):
    """Generate all word windows of a specific size, in the corpus. Implemented as https://github.com/yao8839836/text_gcn/blob/master/build_graph.py#L379

    Args:
        corpus (list): List of documents represented each as tokenized words.
        window_size (int, optional): Window size to use. Defaults to 15.

    Returns:
        list: List of word windows (lists of words)
    """
    windows = []
    for doc in corpus:
        length = len(doc)
        if length <= window_size:
            windows.append(doc)
        else:
            for j in range(length - window_size + 1):
                window = doc[j: j + window_size]
                windows.append(window)
    return windows

def word_frequency(windows):
    """#W(i) is the number of sliding windows in a corpus that contain word i. Implemented as https://github.com/yao8839836/text_gcn/blob/master/build_graph.py#L396

    Args:
        windows (list): Word windows

    Returns:
        dict: Word frequencies
    """
    word_window_freq = defaultdict(lambda: 0)
    for window in windows:
        appeared = set()
        for word in window:
            # We only count every word once inside a window
            if word in appeared:
                continue
            word_window_freq[word] += 1
            appeared.add(word)
    return word_window_freq

#
def word_pair_frequency(windows):
    """#W(i,j) is the number of sliding windows that contain both word i and j.

    Args:
        windows (list): Word windows

    Returns:
        dict: Word pair frequencies
    """
    word_pair_freq = defaultdict(lambda: 0)
    for window in windows:
        window = set(window)
        pairs = combinations(window, 2)
        for pair in pairs:
            word_pair_freq[tuple(sorted(pair))] += 1
    return word_pair_freq

def PMI(corpus, window_size=15):
    """Computes PMI scores.

    Args:
        corpus (list): List of documents represented each as tokenized words.
        window_size (int, optional): Window size to use. Defaults to 15.

    Returns:
        dict: PMI scores for word pairs.
    """
    windows = word_windows(corpus, window_size)
    num_windows = len(windows)
    word_freq = word_frequency(windows)
    word_pair_freq = word_pair_frequency(windows)

    pmi  = {}
    for (word_i, word_j), w_ij in word_pair_freq.items():
        # log((w(i,j)/w) / ((w(i)/w) * (w(j)/w))) = log(w(i,j)) - log(w(i)) - log(w(j)) + log(w)
        w_i = word_freq[word_i]
        w_j = word_freq[word_j]
        score = np.log(w_ij) - np.log(w_i) - np.log(w_j) + np.log(num_windows)
        if score > 0:
            pmi[(word_i, word_j)] = score

    return pmi


# This can be extended to allow tokenization/lowercase if needed
def tf_idf_mtx(corpus):
    """Computes tf.idf scores.

    Args:
        corpus (list): List of documents represented each as tokenized words.

    Returns:
        mtx: Sparse matrix containing tf.idf scores.
        words: List of all words in the order as they appear as columns in the matrix.
    """
    vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda x : x)
    mtx = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()

    return mtx, words