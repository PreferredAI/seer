class SentencePair:
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Init %s" % self.__class__.__name__)

    def compute_cost(self, sentences, pairs=None):
        raise NotImplementedError


class TfIdfSentencePair(SentencePair):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def compute_cost(self, sentences, pairs=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        X = TfidfVectorizer().fit_transform(sentences)
        similarities = (1 + cosine_similarity(X, X)) / 2
        costs = abs(1 - similarities)
        return costs
