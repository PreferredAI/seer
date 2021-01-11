import os
import argparse
from sentence_pair_model import TfIdfSentencePair
from collections import Counter
from explanation_generation import (
    get_corpus,
    get_preference,
    get_coherence_manager,
    get_contextualizer,
    get_generator,
    get_candidates,
)
from local_search_contextualized_opinion import local_search_contextualize_opinion
from util import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="A10L9NQO44OLOU,B0044T2KBU,toy,toy",
        help="Input user, item, demanded aspects (comma seperated). Ex: <userID>,<itemID>,<aspect1>,<aspect2>",
    )
    parser.add_argument(
        "-c",
        "--corpus_path",
        type=str,
        default="data/toy/train.csv",
        help="Input corpus path",
    )
    parser.add_argument(
        "-s",
        "--strategy",
        choices=[
            "greedy-efm",
            "mip-efm",
        ],
        default="greedy-efm",
    )
    parser.add_argument(
        "-p",
        "--preference_dir",
        type=str,
        default="data/toy/efm",
        help="Preference path",
    )
    parser.add_argument(
        "-m", "--contextualizer_path", type=str, default="data/toy/asc2v/model.params"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.5,
        help="Trace off factor between open review cost and representative sentence cost",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return args


def parse_input(input_string):
    input_tokens = input_string.split(",")
    user, item, demanded_aspects = input_tokens[0], input_tokens[1], input_tokens[2:]
    return user, item, Counter(demanded_aspects)


def main(args):
    corpus = get_corpus(args.corpus_path)
    user, item, demanded_aspects = parse_input(args.input)
    candidates, simplified_demand = get_candidates(
        corpus, user, item, demanded_aspects, simplify=True
    )
    print("User:", user)
    print("Item:", item)
    print("Demanded aspects:", demanded_aspects)
    print("Simplify demanded aspects:", simplified_demand)
    print("# Candidate sentences:", len(candidates))
    print("-" * 20)
    preference = get_preference(args.preference_dir, args.strategy, args.verbose)
    sentence_pair_model = TfIdfSentencePair(args.verbose)
    generator = get_generator(
        args.strategy,
        preference,
        sentence_pair_model,
        alpha=args.alpha,
        verbose=args.verbose,
    )
    result = generator.generate(user, item, simplified_demand, candidates)
    selected_sentences = result.get("selected_sentences", [])
    selected_aspects = result.get("selected_aspects", [])
    print("Selected sentences   :", selected_sentences)
    contextualizer = get_contextualizer(
        args.contextualizer_path,
        preference,
        strategy=args.strategy,
        verbose=args.verbose,
    )
    explanations = local_search_contextualize_opinion(
        user,
        item,
        selected_sentences,
        selected_aspects,
        corpus,
        contextualizer,
        sentence_pair_model,
        top_k=args.top_k,
        strategy=args.strategy,
        verbose=args.verbose,
    )
    print("Generated explanation:", explanations)


if __name__ == "__main__":
    main(parse_arguments())