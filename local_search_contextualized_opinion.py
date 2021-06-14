import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from explanation_generation import (contextualize_candidate_sentences,
                                    get_contextualizer, get_preference)
from sentence_pair_model import TfIdfSentencePair
from util import convert_str_to_list, substitute_word

summary_report = {}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="selected.csv",
        help="Selected sentences file path",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        default="data/toy/train.csv",
        help="Corpus file path (train.csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="explanations.csv",
        help="Output file path",
    )
    parser.add_argument(
        "-p",
        "--preference_dir",
        type=str,
        default="data/toy/efm",
        help="EFM/MTER output directory",
    )
    parser.add_argument(
        "-m", "--contextualizer_path", type=str, default="result/model.params"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=10,
        help="Top k opinions for contextualization",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        choices=[
            "greedy-efm",
            "ilp-efm",
        ],
        default="greedy-efm",
        help="Strategy",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_path", type=str, default="debug_local_search.pkl")
    parser.add_argument("--debug_size", type=int, default=100)
    return parser.parse_args()


def compute_representative_cost(sentences, represented_sentences, spm):
    all_sentences = sentences + represented_sentences
    pairs = [
        (i, j)
        for i in range(len(sentences))
        for j in range(len(sentences), len(all_sentences))
    ]
    costs = spm.compute_cost(all_sentences, pairs)
    cost = 0
    for j in range(len(sentences), len(all_sentences)):
        min_cost = min(costs[: len(sentences), j])
        cost += min_cost
    return cost


def local_search_contextualize_opinion(
    user,
    item,
    sentences,
    aspects,
    corpus,
    contextualizer,
    sentence_pair_model,
    top_k=None,
    strategy="ilp-efm",
    verbose=False,
):
    local_searched_sentences = []
    if len(sentences) == 0:
        return local_searched_sentences

    review_idx = "{}-{}".format(user, item)

    candidates = corpus[(corpus["asin"] == item) & (corpus["aspect"].isin(aspects))]
    candidates = contextualize_candidate_sentences(
        candidates, user, contextualizer, top_k=top_k
    )
    if "contextualized" in strategy:
        candidates["instance"] = candidates.apply(
            lambda x: "{}-{}-{}".format(x["asin"], x["aspect"], x["sentence"]), axis=1
        )
    else:
        candidates["instance"] = candidates.apply(
            lambda x: "{}-{}-{}".format(x["asin"], x["aspect"], x["original sentence"]),
            axis=1,
        )
        candidates["sentence"] = candidates["original sentence"]
    candidates.drop_duplicates("instance", inplace=True)
    candidates = candidates.set_index(["instance"])
    aspect_sentences_map = {}
    for aspect, sentence in zip(candidates["aspect"], candidates["sentence"]):
        aspect_sentences = aspect_sentences_map.setdefault(aspect, [])
        if sentence not in sentences:
            aspect_sentences.append(sentence)

    solution = {}
    for aspect, sentence in zip(aspects, sentences):
        aspect_sentences = solution.setdefault(aspect, [])
        aspect_sentences.append(sentence)

    for aspect, sentence in zip(aspects, sentences):
        represented_sentences = aspect_sentences_map.get(aspect)
        if len(represented_sentences) > 0:
            solution_sentences = solution[aspect].copy()
            instance = candidates.loc["{}-{}-{}".format(item, aspect, sentence)]
            predicted_opinions = instance["top k opinions"]
            opinion_position = instance["opinion_pos"]

            # do local search here
            best_opinion = sentence.split()[opinion_position]  # raw opinion
            new_sentence = sentence
            best_cost = compute_representative_cost(
                solution_sentences, represented_sentences, sentence_pair_model
            )
            solution_sentences.remove(sentence)
            best_idx = -1
            for idx, opinion in enumerate(predicted_opinions):
                new_sentence = substitute_word(sentence, opinion, opinion_position)
                temp_solution_sentences = solution_sentences + [new_sentence]
                cost = compute_representative_cost(
                    temp_solution_sentences, represented_sentences, sentence_pair_model
                )
                if cost < best_cost:
                    best_idx = idx
                    best_cost = cost
                    best_opinion = opinion
            sentence = substitute_word(sentence, best_opinion, opinion_position)
            summary_report.setdefault(review_idx, []).append(best_idx)
        local_searched_sentences.append(sentence)
    return local_searched_sentences


if __name__ == "__main__":
    args = parse_arguments()
    print("strategy: %s" % args.strategy)
    print("load input from %s" % args.input)
    df = pd.read_csv(args.input)
    df = df[df["selected sentences"].notnull()]
    if args.debug:
        if args.debug_size > 0:
            df = df[: args.debug_size]
    df["selected sentences"] = df["selected sentences"].apply(
        lambda x: convert_str_to_list(x)
    )

    print("load corpus from %s" % args.corpus)
    corpus = pd.read_csv(args.corpus)

    preference = get_preference(args.preference_dir, args.strategy, args.verbose)

    contextualizer = get_contextualizer(
        args.contextualizer_path, preference, args.strategy, verbose=args.verbose
    )

    sentence_pair_model = TfIdfSentencePair(
        args.verbose,
    )

    df["backup sentences"] = df["sentences"]

    tqdm.pandas(desc="Local search")

    df["selected sentences"] = df.progress_apply(
        lambda row: local_search_contextualize_opinion(
            row["reviewerID"],
            row["asin"],
            row["selected sentences"],
            str(row["aspects"]).split(","),
            corpus,
            contextualizer,
            sentence_pair_model,
            top_k=args.top_k,
            strategy=args.strategy,
            verbose=args.verbose,
        ),
        axis=1,
    )

    df["sentences"] = df["selected sentences"].apply(lambda x: " . ".join(x))

    df.to_csv(args.out, index=False)
    if args.debug:
        import pickle

        with open(args.debug_path, "wb") as f:
            pickle.dump(summary_report, f)
        print(summary_report)
