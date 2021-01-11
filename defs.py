import os


class Toks:
    UNK = 0
    BOS = 1  # beginning of sentence
    EOS = 2  # end of sentence


UNK_TOKEN = "<UNK>"

NEGATIVE_SAMPLING_NUM = 10
IN_TO_OUT_UNITS_RATIO = 2
MAX_ORDER_BLEU = 2
LOG_FILENAME = "log"
TRAIN_FILENAME = "train.csv"
VALIDATION_FILENAME = "validation.csv"
TEST_FILENAME = "test.csv"
SENTIRES_DIR = "sentires"
SPLIT_DIR = "split"
REVIEW_DIR = "review"
GROUPED_DIR = "grouped_by_item"
FILTERED_FILE = "filtered.csv"
SPLITTED_TRAIN_FILE = os.path.join(SPLIT_DIR, TRAIN_FILENAME)
SPLITTED_VALIDATION_FILE = os.path.join(SPLIT_DIR, VALIDATION_FILENAME)
SPLITTED_TEST_FILE = os.path.join(SPLIT_DIR, TEST_FILENAME)
REVIEW_TRAIN_FILE = os.path.join(REVIEW_DIR, TRAIN_FILENAME)
REVIEW_VALIDATION_FILE = os.path.join(REVIEW_DIR, VALIDATION_FILENAME)
REVIEW_TEST_FILE = os.path.join(REVIEW_DIR, TEST_FILENAME)
USER_DICT_FILENAME = "user.dic"
ITEM_DICT_FILENAME = "item.dic"
ASPECT_DICT_FILENAME = "aspect.dic"
OPINION_DICT_FILENAME = "opinion.dic"
MODEL_FILENAME = "model"
VOCAB_FILENAME = "vocab.dic"
SC_ASPECT_DICT_FILENAME = "aspects.dic"
SC_OPINION_DICT_FILENAME = "opinions.dic"
SC_WORD_COUNTS_FILENAME = "words.cnt"
ASPECT_OPINION_FILENAME = "aspect_opinion.json"
UIF_TRAIN_FILENAME = "uif.train"
UIF_TEST_FILENAME = "uif.test"
UIFO_TRAIN_FILENAME = "uifo.train"
UIFO_TEST_FILENAME = "uifo.test"
SENT_COUNTS_FILENAME = "s_counts"
WORD_COUNTS_FILENAME = "w_counts"
ASPECT_COUNTS_FILENAME = "a_counts"
TOTAL_COUNTS_FILENAME = "totals"
OPINION_COUNTS_FILENAME = "o_counts"
DATA_FIELD_NAMES = [
    "reviewerID",
    "asin",
    "overall",
    "score",
    "aspect_pos",
    "aspect",
    "opinion_pos",
    "opinion",
    "sentence",
    "sentence_len",
    "count",
]
RANKING_FILENAME = "rankings.json"
FULL_TEST_RESULTS_FILENAME = "full.json"
RESULT_FILENAME = "results.csv"
EXPLANATION_OUTPUT_FILENAME = "explanations.csv"
EXPLANATION_EVAL_FILENAME = "explanations_eval.csv"
EXPLANATION_SUMMARY_FILENAME = "explanations_summary.csv"
SENTENCE_SELECTION_OUTPUT_FILENAME = "sentence_selection_result.csv"
SENTENCE_SELECTION_EVAL_FILENAME = "sentence_selection_eval.csv"
SENTENCE_SELECTION_SUMMARY_FILENAME = "sentence_selection_summary.csv"
SOURCE_ASPECT_SCORE = {
    "aoc2v": "overall",
    "sc2v": "aspect_score_efm",
    "asc2v": "aspect_score_efm",
    "sc2v-mter": "aspect_score_mter",
    "asc2v-mter": "aspect_score_mter",
}
