import sys
import gzip
import re
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from math import exp
import csv
import json
from ast import literal_eval


OPINION_EXP = re.compile(r"(.*)<o>(.*?)</o>(.*)")
ASPECT_EXP = re.compile(r"(.*)<f>(.*?)</f>(.*)")
TAGGED_EXP = re.compile(r"<\w>(.*?)</\w>")
TARGET_EXP = re.compile(r"\[.*\]")


def readline_gzip(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield line


def readline(path):
    with open(path, "r") as f:
        for line in f:
            yield line


def unique(sequence):
    """
    Returns a unique list preserve the order of original list
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def to_dict(values):
    value2index = {}
    for i, item in enumerate(unique(values)):
        value2index[item] = i
    return value2index


def save_dict(value2index, path):
    with open(path, "w") as f:
        for value, index in value2index.items():
            f.write("%s %d\n" % (value, index))
    return value2index


def load_dict(path, sep=None):
    dic = {}
    with open(path, "r") as f:
        for line in f:
            try:
                [item, index] = line.split(sep)
                dic[item] = int(index)
            except:
                print("WARN - skipping invalid line: {}".format(line), sys.exc_info())
    return dic


def save_count(values, path):
    counts = Counter(values)
    with open(path, "w") as f:
        for w, count in counts.most_common():
            f.write("%s %d\n" % (w, count))
    return counts


def load_count(path, sep=None, dtypeKey=""):
    counts = Counter()
    with open(path, "r") as f:
        for line in f:
            try:
                [w, count] = line.strip().split(sep)
                if dtypeKey == "int":
                    w = int(w)
                counts[w] = int(count)
            except:
                print("WARN - skipping invalid line: {}".format(line), sys.exc_info())
    return counts


def reverse_key(key_value):
    return {v: k for k, v in key_value.items()}


def parse_sentence(sentence, opinion, aspect):
    stemmer = PorterStemmer()
    sentence = re.sub(
        re.compile("(^| )({})".format(opinion)), r"\1<o>\2</o>", sentence, 1
    )
    if not OPINION_EXP.match(sentence):
        sentence = re.sub(
            re.compile("(^| )({})".format(stemmer.stem(opinion))),
            r"\1<o>\2</o>",
            sentence,
            1,
        )
    sentence = re.sub(
        re.compile("(^| )({})".format(aspect)), r"\1<f>\2</f>", sentence, 1
    )
    if not ASPECT_EXP.match(sentence):
        sentence = re.sub(
            re.compile("(^| )({})".format(stemmer.stem(aspect))),
            r"\1<f>\2</f>",
            sentence,
            1,
        )
    sentence = re.sub(
        re.compile("<o>{}</o>".format(opinion)),
        "<o>{}</o>".format("_".join(opinion.split(" "))),
        sentence,
    )
    sentence = re.sub(
        re.compile("<f>{}</f>".format(aspect)),
        "<f>{}</f>".format("_".join(aspect.split(" "))),
        sentence,
    )
    sentence = re.sub(r"(<\w?>[ \w]+)(</\w?>)([-\w]+)", r"\1\3\2", sentence)
    sentence = re.sub(r"\(\d+\)$", "", sentence).strip().lower()

    opinion_pos = None
    aspect_pos = None

    opinion_segments = OPINION_EXP.match(sentence)
    if opinion_segments is not None:
        opinion_pos = len(
            word_tokenize(re.sub(TAGGED_EXP, r"\1", opinion_segments.group(1)))
        )
        opinion = opinion_segments.group(2)
    aspect_segments = ASPECT_EXP.match(sentence)
    if aspect_segments is not None:
        aspect_pos = len(
            word_tokenize(re.sub(TAGGED_EXP, r"\1", aspect_segments.group(1)))
        )
        aspect = aspect_segments.group(2)
    tokens = word_tokenize(re.sub(TAGGED_EXP, r"\1", sentence))
    sentence_len = len(tokens)
    sentence = " ".join(tokens)
    return sentence, sentence_len, opinion_pos, opinion, aspect_pos, aspect


def to_one_hot(idx, size, value=1.0):
    one_hot = np.zeros(size).astype(np.float32)
    one_hot[int(float(idx))] = value
    return one_hot


def flatten_json(json_content):
    csv_content = {}
    for k, v in json_content.items():
        if not isinstance(v, dict):
            csv_content[k] = v
        else:
            for k1, v1 in v.items():
                csv_content["{}_{}".format(k, k1)] = v1
    return csv_content


def dict_to_csv(json_content, path):
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(json_content.keys()))
        writer.writeheader()
        writer.writerow(json_content)


def dump_json(json_content, path):
    with open(path, "w") as f:
        json.dump(json_content, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def export_spare_matrix(M, path, sep="\t"):
    assert len(M.shape) == 2
    (d1, d2) = M.shape
    with open(path, "w") as f:
        f.write("{}\t{}\t{}\n".format(d1, d2, np.count_nonzero(M)))
        for i in range(d1):
            for j in range(d2):
                if M[i, j] != 0:
                    f.write("{}\t{}\t{}\n".format(i, j, M[i, j]))


def export_dense_matrix(M, path):
    assert len(M.shape) == 2
    (d1, d2) = M.shape
    with open(path, "w") as f:
        f.write("Dimension: {} x {}\n".format(d1, d2))
        for i in range(d1):
            f.write("[{}]\n".format("\t".join([str(j) for j in M[i]])))


def load_sparse_matrix(path):
    with open(path, "r") as f:
        line = f.readline()
        tokens = line.strip().split()
        assert len(tokens) == 3
        r, c, n = int(tokens[0]), int(tokens[1]), int(tokens[2])
        matrix = np.zeros((r, c))
        for i in range(n):
            line = f.readline()
            tokens = line.strip().split()
            assert len(tokens) == 3
            matrix[int(tokens[0])][int(tokens[1])] = float(tokens[2])
    return matrix


def load_dense_matrix(path):
    result = []
    with open(path, "r") as f:
        tokens = f.readline().split(":")[1].split("x")
        assert len(tokens) == 2
        r, c = int(tokens[0]), int(tokens[1])
        for i in range(r):
            tokens = f.readline().strip()[1:-1].split()
            assert len(tokens) == c
            values = [float(v) for v in tokens]
            result.append(values)
    return np.array(result)


def export_dense_tensor(T, path):
    assert len(T.shape) == 3
    (d1, d2, d3) = T.shape
    with open(path, "w") as f:
        f.write("Dimension: {} x {} x {}\n".format(d1, d2, d3))
        for i in range(d1):
            f.write(
                "{}\n".format(
                    ",".join(
                        ["[{}]".format("\t".join([str(k) for k in j])) for j in T[i]]
                    )
                )
            )


def load_dense_tensor(path):
    result = []
    with open(path, "r") as f:
        tokens = f.readline().split(":")[1].split("x")
        assert len(tokens) == 3
        d1, d2, d3 = int(tokens[0]), int(tokens[1]), int(tokens[2])
        for i in range(d1):
            lst = f.readline().strip().split(",")
            arr = []
            for j in range(d2):
                values = [float(v) for v in lst[j][1:-1].split()]
                arr.append(values)
            result.append(arr)
        return np.array(result)


def empty_file(path):
    with open(path, "w") as f:
        f.write("")


def frequent_score(cnt, N):
    return 1 + (N - 1) * (2 / (1 + exp(-cnt)) - 1)


def sentiment_score(sentiment, N):
    return 1 + (N - 1) / (1 + exp(-sentiment))


def lcs(a, b):
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the subsequence out from the matrix
    result = []
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result.append(a[x - 1])
            x -= 1
            y -= 1
    return result[::-1]


def array2string(x):
    assert len(np.shape(x)) <= 2
    if len(np.shape(x)) == 1:
        return ",".join([str(i) for i in x])
    elif len(np.shape(x)) == 2:
        return ";".join([array2string(i) for i in x])


def string2array(x):
    if len(x.split(";")) > 1:
        return [[j for j in i.split(",")] for i in x.split(";")]
    return [i for i in x.split(",")]


def substitute_word(sentence, new_word, position):
    sentence = sentence.split()
    sentence[position] = new_word
    return " ".join(sentence)


def convert_str_to_list(cell):
    return literal_eval(cell)
