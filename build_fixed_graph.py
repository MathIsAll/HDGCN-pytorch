import pickle as pkl
import numpy as np
from utils import clean_str
import scipy.sparse as sp
from tqdm import tqdm
from utils import clean_str
import torch

word_embeddings = dict()
with open('glove.840B.300d.txt', 'r') as f:
    for line in f.readlines():
        data = line.split(' ')
        word_embeddings[str(data[0])] = list(map(float, data[1:]))
word_embedding_dim = 300

docs = open("data/mr/mr.clean.txt", 'r')
doc_list = docs.readlines()
docs.close()

# build vocab
word_set = set()
for doc_words in doc_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
word_set.add("<pad>")

vocab = list(word_set)
vocab_size = len(vocab)
oov = np.random.uniform(-0.01, 0.01, (vocab_size, word_embedding_dim))

labels = open("data/mr/mr.txt", 'r')
label_list = labels.readlines()
labels.close()

labels = []
for i in range(len(doc_list)):
    labels.append(clean_str(label_list[i]).split()[-1])
label_set = set(labels)
label_set = list(label_set)

"""
def dependency_adj_matrix(text, window_size=11, weighted_graph=False):
    # https://spacy.io/docs/usage/processing-text
    doc_len = len(text)

    # sliding windows
    windows = []
    if doc_len <= window_size:
        windows.append(text)
    
        for i in range(doc_len - window_size + 1):
            window = text[i: i + window_size]
            windows.append(window)
    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_id_map[word_p]
                word_q = window[q]
                word_q_id = word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
    row = []
    col = []
    weight = []
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        row.append(word_id_map[vocab[p]])
        col.append(word_id_map[vocab[q]])
        weight.append(word_pair_count[key] if weighted_graph else 1.)
    for key in range(vocab_size):
        row.append(key)
        col.append(key)
        weight.append(1.)
    adj = sp.csr_matrix((weight, (row, col)), shape=(vocab_size, vocab_size))
    adj = adj.tocoo()
    adj = torch.sparse.FloatTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                                   torch.FloatTensor(adj.data.astype(np.float)))
    # adj = normalize_adj(adj)

    #matrix = np.zeros((vocab_size, vocab_size)).astype('float32')
    #for row in range(doc_len):
    #    for tk in range(doc_len):
    #        matrix[row, tk] += adj[word_id_map[orig_doc[row]], word_id_map[orig_doc[tk]]]
    return adj
"""


def dependency_adj_matrix(text, max_len, window_size=3, weighted_graph=False):
    # https://spacy.io/docs/usage/processing-text
    doc_len = len(text)
    doc_vocab = list(set(text))
    vocab_size = len(doc_vocab)
    word_ids_map = {}
    ids_word_map = {}
    for j in range(vocab_size):
        word_ids_map[doc_vocab[j]] = j
        ids_word_map[j] = doc_vocab[j]

    # sliding windows
    windows = []
    if doc_len <= window_size:
        windows.append(text)
    else:
        for i in range(doc_len - window_size + 1):
            window = text[i: i + window_size]
            windows.append(window)
    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_ids_map[word_p]
                word_q = window[q]
                word_q_id = word_ids_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
    row = []
    col = []
    weight = []
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        row.append(word_ids_map[doc_vocab[p]])
        col.append(word_ids_map[doc_vocab[q]])
        weight.append(word_pair_count[key] if weighted_graph else 1.)
    for key in range(vocab_size):
        row.append(key)
        col.append(key)
        weight.append(1.)
    adj = sp.csr_matrix((weight, (row, col)), shape=(vocab_size, vocab_size))

    matrix = np.zeros((max_len, max_len)).astype('float32')
    for row in range(doc_len):
        for tk in range(doc_len):
            matrix[row, tk] += adj[word_ids_map[text[row]], word_ids_map[text[tk]]]
    matrix = normalize_adj(matrix)
    return matrix


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


all_graphs = []
all_labels = []
all_features = []
all_sentences = []
max_seq_len = 35
for i in tqdm(range(len(doc_list))):
    doc = doc_list[i].lower().strip()
    # doc = clean_str(doc[0]) + " " + clean_str(doc[1])
    doc = doc.split()[:max_seq_len]
    doc_words = ["<pad>" for _ in range(max_seq_len)]
    doc_words[:len(doc)] = doc[:]
    features = []
    for key in range(len(doc_words)):
        if doc_words[key] in word_embeddings:
            features.append(word_embeddings[doc_words[key]])
        else:
            features.append(oov[vocab.index(doc_words[key]), :])
    features = np.array(features)
    adj_matrix = dependency_adj_matrix(doc, max_len=max_seq_len)
    one_hot = [0 for l in range(len(label_set))]
    label_index = label_set.index(labels[i])
    one_hot[label_index] = 1
    all_features.append(features)
    all_graphs.append(adj_matrix)
    all_labels.append(np.array(one_hot))
    sentence = []
    for k in range(max_seq_len):
        sentence.append(vocab.index(doc_words[k]))
    all_sentences.append(sentence)

with open("data/mr/mr.all.features", 'wb') as f:
    pkl.dump(all_features, f)

with open("data/mr/mr.all.adj", 'wb') as f:
    pkl.dump(all_graphs, f)

with open("data/mr/mr.all.label", 'wb') as f:
    pkl.dump(all_labels, f)

with open("data/mr/mr.all.sentence", 'wb') as f:
    pkl.dump(all_sentences, f)

with open("data/mr/mr.all.vocab", 'wb') as f:
    pkl.dump(vocab, f)
