from sklearn.metrics.pairwise import cosine_similarity


class CoherenceManager:
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Init %s" % self.__class__.__name__)

    def compute_cost(self, user, item, reviewer):
        raise NotImplementedError()


class DummyCoherenceManager(CoherenceManager):
    def compute_cost(self, user, item, reviewer):
        """Dummy cost = 1"""
        return 1


class ContextualizedCoherenceManager(CoherenceManager):
    def __init__(self, preference, verbose=False):
        super().__init__(verbose)
        self.preference = preference

    def compute_cost(self, user, item, reviewer):
        user_aspect_vector = self.preference.get_aspect_vector(user, item).reshape(
            1, -1
        )
        reviewer_aspect_vector = self.preference.get_aspect_vector(
            reviewer, item
        ).reshape(1, -1)

        score = (
            1 + cosine_similarity(user_aspect_vector, reviewer_aspect_vector).sum()
        ) / 2
        return abs(1 - score)
