import time
from collections import namedtuple

import numpy as np

from seer_ilp import Company, Customer, Facility, FacilityTypeDemand, solve_seer_ilp

Sentence = namedtuple("Sentence", ("name", "review", "aspect", "text"))
Review = namedtuple("Review", ("name", "reviewer", "cost"))
AspectDemand = namedtuple("AspectDemand", ("aspect", "demand"))
DUMMY_ASPECT = "<DUM>"

EPS = 1e-9


class ExplanationGenerator:
    def __init__(self, verbose=False):
        self.verbose = verbose
        if self.verbose:
            print("Init %s" % self.__class__.__name__)

    def generate(self, user=None, item=None, demand=None, candidates=None):
        raise NotImplementedError


class SentenceSelector(ExplanationGenerator):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)


class TextRankSentenceSelector(SentenceSelector):
    def __init__(self, verbose=False):
        super().__init__(verbose)

    def generate(self, demand, candidates):
        from summa import summarizer
        result = {"demand": demand, "candidates": candidates}
        if len(candidates) > 0:
            cur_time = time.time()

            sentences = candidates["sentence"].tolist()
            totalDemand = sum(demand.values())
            totalSentences = len(sentences)
            ratio = 1.0 * totalDemand / totalSentences + EPS
            summary = summarizer.summarize(" . ".join(sentences), ratio=ratio)
            sentences = [s.strip() for s in summary.strip().split(".") if len(s) > 0]

            result["selected_sentences"] = []
            result["selected_aspects"] = []
            result["selected_reviews"] = []
            for _, row in candidates.iterrows():
                if row["sentence"] in sentences:
                    if result["selected_sentences"].count(
                        row["sentence"]
                    ) >= sentences.count(row["sentence"]):
                        continue
                    result["selected_sentences"].append(row["sentence"])
                    result["selected_aspects"].append(row["aspect"])
                    if row["id"] not in result["selected_reviews"]:
                        result["selected_reviews"].append(row["id"])
            result["solve_time"] = time.time() - cur_time

        return result


class ILPSentenceSelector(SentenceSelector):
    def __init__(
        self,
        coherence_manager,
        sentence_pair_model,
        alpha=1.0,
        strategy="ilp-efm",
        verbose=False,
    ):
        super().__init__(verbose)
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha
        self.coherence_manager = coherence_manager
        self.sentence_pair_model = sentence_pair_model
        self.strategy = strategy

    def _preprocessing(self, user, item, candidates, demand):
        facilityDemands = [
            FacilityTypeDemand(aspect, count, None) for aspect, count in demand.items()
        ]
        # construct facilities & customers
        facilities = []
        customers = []
        reviews = set()
        for _, row in candidates.iterrows():
            record = (
                row["index"],
                row["aspect"],
                row["id"],
                row["sentence"],
                (int(row["opinion_pos"]), int(row["aspect_pos"])),
            )
            facilities.append(Facility(*record))
            customers.append(Customer(*record))
            reviews.add((row["id"], row["reviewerID"], row["asin"]))

        costs = self._compute_cost(facilities)

        companies = []
        for reviewID, reviewer, item in reviews:
            cost = self.coherence_manager.compute_cost(user, item, reviewer)
            companies.append(Company(reviewID, cost, None))
        return companies, facilities, customers, costs, facilityDemands

    def _compute_cost(self, facilities):
        """
        Number of customers = number of facilities
        """
        corpus = [facility.content for facility in facilities]
        type2ids = {}
        for idx, facitity in enumerate(facilities):
            ids = type2ids.setdefault(facitity.type, [])
            ids.append(idx)
        pairs = [
            (i, j) for ids in type2ids.values() for i in ids for j in ids if i != j
        ]
        return self.sentence_pair_model.compute_cost(corpus, pairs)

    def generate(self, user, item, demand, candidates):
        result = {"demand": demand, "candidates": candidates}

        if len(candidates) > 0:
            (
                companies,
                facilities,
                customers,
                costs,
                facilityDemands,
            ) = self._preprocessing(user, item, candidates, demand)
            log_verbose = "Normal" if self.verbose else "Quiet"
            solution, selectedFacilities, selectedCompanies = solve_seer_ilp(
                companies,
                facilities,
                customers,
                facilityDemands,
                costs,
                self.alpha,
                log_verbose,
            )

            result["selected_aspects"] = [
                facilities[i].type for i in selectedFacilities
            ]
            result["objective_value"] = (
                solution.get_objective_values()[0]
                if solution.get_objective_values() is not None
                else None
            )
            result["objective_bound"] = (
                solution.get_objective_bounds()[0]
                if solution.get_objective_bounds() is not None
                else None
            )
            result["objective_gap"] = (
                solution.get_objective_gaps()[0]
                if solution.get_objective_gaps() is not None
                else None
            )
            result["selected_sentences"] = [
                facilities[i].content for i in selectedFacilities
            ]
            result["selected_reviews"] = [companies[i].name for i in selectedCompanies]
            result["solve_time"] = solution.get_solve_time()

        return result


class GreedySentenceSelector(ILPSentenceSelector):
    def __init__(
        self,
        coherence_manager,
        sentence_pair_model,
        alpha,
        strategy="greedy",
        verbose=False,
    ):
        super().__init__(
            coherence_manager, sentence_pair_model, alpha, strategy, verbose
        )

    def greedy_select_sentences(self, reviews, sentences, costs, aspect_demand):
        """
        review(id, reviewer, coherence cost)
        sentence(id, review_id, aspect, text)
        demand(aspect, demand)
        """
        start_time = time.time()

        empty_indices = np.empty(0).astype(int)
        selected_sentences = np.empty(0).astype(int)
        selected_reviews = np.empty(0).astype(int)

        n_review = len(reviews)
        n_sentence = len(sentences)
        n_aspect = len(aspect_demand)
        non_represented_sentences = np.array(range(n_sentence)).astype(int)
        all_sentences = np.array(range(n_sentence)).astype(int)
        all_reviews = np.array(range(n_review)).astype(int)
        aspects = np.array(range(n_aspect)).astype(int)
        demands = np.array([aspect.demand for aspect in aspect_demand]).astype(int)
        review_costs = np.array([self.alpha * reviews[i].cost for i in range(n_review)])

        costs = (1 - self.alpha) * costs
        review2id = {review.name: idx for idx, review in enumerate(reviews)}
        aspect2id = {x.aspect: idx for idx, x in enumerate(aspect_demand)}

        review_sentences = np.zeros((n_review, n_sentence)).astype(int)
        aspect_sentences = np.zeros((n_aspect, n_sentence)).astype(int)

        for i in range(n_sentence):
            review_sentences[review2id[sentences[i].review], i] = 1
            aspect_sentences[aspect2id[sentences[i].aspect], i] = 1

        def get_aspect(sentence_idx):
            return np.where(aspect_sentences[:, sentence_idx] == 1)[0][0]

        def get_review(sentence_idx):
            return np.where(review_sentences[:, sentence_idx] == 1)[0][0]

        def get_aspect_sentences(aspect_idx):
            return np.where(aspect_sentences[aspect_idx] == 1)[0].astype(int)

        def get_review_sentences(review_idx):
            return np.where(review_sentences[review_idx] == 1)[0].astype(int)

        def get_same_aspect_sentences(sentence_idx):
            aspect_idx = get_aspect(sentence_idx)
            return get_aspect_sentences(aspect_idx)

        def intersect(arr1, arr2, assume_unique=True):
            return np.intersect1d(arr1, arr2, assume_unique=assume_unique).astype(int)

        def append(arr1, arr2):
            return np.append(arr1, arr2).astype(arr1.dtype)

        def union(arr1, arr2):
            return np.union1d(arr1, arr2).astype(int)

        def diff(arr1, arr2, assume_unique=True):
            return np.setdiff1d(arr1, arr2, assume_unique=assume_unique).astype(int)

        def get_min_r_cost(selected_sids, sid):
            return costs[
                intersect(selected_sids, get_same_aspect_sentences(sid)), sid
            ].min()

        def get_cost(selected_rids, selected_sids, represented_sids):
            c_cost = review_costs[selected_rids].sum()
            r_cost = sum(
                [get_min_r_cost(selected_sids, sid) for sid in represented_sids]
            )
            return c_cost + r_cost

        def get_avg_cost(selected_rids, selected_sids, represented_sids):
            return get_cost(selected_rids, selected_sids, represented_sids) / len(
                represented_sids
            )

        while len(non_represented_sentences) > 0:
            selected = {"min_avg_cost": 10e10}
            avail_reviews = diff(all_reviews, selected_reviews)

            for rid in avail_reviews:
                sids_by_review = get_review_sentences(rid)
                avail_sentences = intersect(sids_by_review, non_represented_sentences)
                if len(avail_sentences) == 0:
                    # sentences of this review is already represented
                    continue
                enum_costs = np.empty(0)
                enum_sentences = empty_indices
                for r_sid in avail_sentences:
                    enum_sentences = append(enum_sentences, r_sid)
                    representative_sentences = intersect(
                        non_represented_sentences, get_same_aspect_sentences(r_sid)
                    )
                    cost = get_cost(
                        empty_indices,
                        union(empty_indices, r_sid),
                        representative_sentences,
                    )
                    enum_costs = append(enum_costs, cost)

                tau_i = empty_indices
                for r_sid in enum_sentences[np.argsort(enum_costs)]:
                    aid = get_aspect(r_sid)
                    aspect_cnt = len(intersect(tau_i, get_aspect_sentences(aid)))
                    if aspect_cnt < demands[aid]:
                        tau_i = union(tau_i, r_sid)

                fixed_representative_sentences = tau_i
                for aid in aspects[demands > 0]:
                    aspect_cnt = len(intersect(tau_i, get_aspect_sentences(aid)))
                    if aspect_cnt == demands[aid]:
                        representative_sentences = intersect(
                            non_represented_sentences, get_aspect_sentences(aid)
                        )
                        fixed_representative_sentences = union(
                            fixed_representative_sentences, representative_sentences
                        )

                avg_cost = get_avg_cost(
                    union(empty_indices, rid), tau_i, fixed_representative_sentences
                )

                if avg_cost < selected["min_avg_cost"]:
                    selected["min_avg_cost"] = avg_cost
                    selected["review"] = rid
                    selected["sentences"] = tau_i
                    selected["represented"] = fixed_representative_sentences

                enum_costs = np.empty(0)
                enum_sentences = empty_indices
                remaining_representative_sentences = diff(
                    non_represented_sentences, fixed_representative_sentences
                )
                for sid in remaining_representative_sentences:
                    r_cost = get_min_r_cost(tau_i, r_sid)
                    enum_sentences = append(enum_sentences, r_sid)
                    enum_costs = append(enum_costs, r_cost)

                covered = fixed_representative_sentences
                for sid in enum_sentences[np.argsort(enum_costs)]:
                    covered = union(covered, sid)
                    avg_cost = get_avg_cost(union(empty_indices, rid), tau_i, covered)

                    if avg_cost > selected["min_avg_cost"]:
                        break
                    elif avg_cost < selected["min_avg_cost"]:
                        selected["min_avg_cost"] = avg_cost
                        selected["review"] = rid
                        selected["sentences"] = tau_i
                        selected["represented"] = covered

            non_represented_sentences = diff(
                non_represented_sentences, selected.get("represented", [])
            )
            selected_sentences = union(
                selected_sentences, selected.get("sentences", [])
            )
            selected_reviews = union(selected_reviews, selected.get("review", []))
            for sid in selected.get("sentences", []):
                demands[get_aspect(sid)] -= 1

        if "local" in self.strategy and sum(demands) > 0:
            while sum(demands) > 0:
                for aid in range(n_aspect):
                    while demands[aid] > 0:
                        avail_sentences = diff(
                            get_aspect_sentences(aid), selected_sentences
                        )

                        best_delta_cost = 10e10
                        promoted_sentence = None
                        promoted_review = None
                        for sid in avail_sentences:
                            aid = get_aspect(sid)
                            delta_cost = -get_min_r_cost(
                                intersect(
                                    get_aspect_sentences(aid), selected_sentences
                                ),
                                sid,
                            )

                            rid = get_review(sid)
                            if rid not in selected_reviews:
                                delta_cost += review_costs[rid]

                            for r_sid in get_same_aspect_sentences(sid):
                                prev_r_cost = get_min_r_cost(
                                    intersect(
                                        get_same_aspect_sentences(r_sid),
                                        selected_sentences,
                                    ),
                                    r_sid,
                                )
                                curr_r_cost = get_min_r_cost(
                                    intersect(
                                        get_same_aspect_sentences(r_sid),
                                        union(selected_sentences, r_sid),
                                    ),
                                    r_sid,
                                )
                                if curr_r_cost < prev_r_cost:
                                    delta_cost += prev_r_cost - curr_r_cost

                            if delta_cost < best_delta_cost:
                                best_delta_cost = delta_cost
                                promoted_sentence = sid
                                promoted_review = rid
                        selected_sentences = union(
                            selected_sentences, promoted_sentence
                        )
                        selected_reviews = union(selected_reviews, promoted_review)

                        demands[aid] -= 1

        total_cost = get_cost(selected_reviews, selected_sentences, all_sentences)
        solve_time = time.time() - start_time
        solution = {"objective_value": total_cost, "solve_time": solve_time}
        return solution, selected_sentences, selected_reviews

    def _prepare_data(self, user, item, candidates, demand):
        aspect_demand = [
            AspectDemand(aspect, count) for aspect, count in demand.items()
        ]

        sentences = []
        review_set = set()  # avoid duplication
        reviews = []
        corpus = []
        for _, row in candidates.iterrows():
            sentences.append(
                Sentence(row["index"], row["id"], row["aspect"], row["sentence"])
            )
            corpus.append(row["sentence"])

            if row["id"] in review_set:
                continue

            review_set.add(row["id"])
            cost = self.coherence_manager.compute_cost(user, item, row["reviewerID"])
            reviews.append(Review(row["id"], row["reviewerID"], cost))

        aspect2ids = {}
        for idx, sentence in enumerate(sentences):
            aspect2ids.setdefault(sentence.aspect, []).append(idx)

        pairs = [
            (i, j) for ids in aspect2ids.values() for i in ids for j in ids if i != j
        ]

        costs = self.sentence_pair_model.compute_cost(corpus, pairs)

        return reviews, sentences, costs, aspect_demand

    def generate(self, user, item, demand, candidates):
        result = {"demand": demand, "candidates": candidates}
        if len(candidates) > 0:
            reviews, sentences, costs, aspect_demand = self._prepare_data(
                user, item, candidates, demand
            )

            (
                solution,
                selected_sentences,
                selected_reviews,
            ) = self.greedy_select_sentences(reviews, sentences, costs, aspect_demand)

            result["selected_aspects"] = [
                sentences[i].aspect for i in selected_sentences
            ]
            result["selected_sentences"] = [
                sentences[i].text for i in selected_sentences
            ]
            result["selected_reviews"] = [reviews[i].name for i in selected_reviews]

            result["solve_time"] = solution.get("solve_time")
            result["objective_value"] = solution.get("objective_value")
        return result
