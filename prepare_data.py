import os
import sys
import argparse
import pandas as pd
import csv
from tqdm import tqdm
from util import save_dict, to_dict, save_count, load_count
from collections import Counter
from defs import *

SPLITTED_INFO_FILENAME = "split.info"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default="data/toy/profile.csv", help="Input data file"
    )
    parser.add_argument("-o", "--out", default="data/toy", help="Ouput directory")
    parser.add_argument(
        "-rv",
        "--ratio_validation",
        type=float,
        default=0.2,
        help="Validation set ratio",
    )
    parser.add_argument(
        "-rt", "--ratio_test", type=float, default=0.2, help="Test set ratio"
    )
    args = parser.parse_args()
    print("Input file:", args.input)
    assert (
        args.ratio_test > 0
        and args.ratio_validation > 0
        and 1.0 - args.ratio_test - args.ratio_validation > 0
    )
    print(
        "Split ratio:",
        round(1 - args.ratio_validation - args.ratio_test, 2),
        args.ratio_validation,
        args.ratio_test,
    )
    print("Output dir:", args.out)
    print("")
    return args

def main(args):
    splitted_path = os.path.join(args.out, SPLIT_DIR)
    if not os.path.exists(splitted_path):
        os.makedirs(splitted_path)

    profile = pd.read_csv(args.input)
    profile["id"] = (
        profile["reviewerID"].map(str)
        + "-"
        + profile["asin"].map(str)
        + "-"
        + profile["unixReviewTime"].map(str)
    )
    reviews = (
        profile.groupby(["reviewerID", "asin", "unixReviewTime", "id"])[
            ["reviewerID", "asin", "unixReviewTime", "id"]
        ]
        .nunique()
        .drop(columns=["reviewerID", "asin", "unixReviewTime", "id"])
        .reset_index()
        .sort_values(by=["reviewerID", "unixReviewTime"], ascending=False)
        .reset_index(drop=True)
        .reset_index()
    )
    maxIdx = reviews.groupby(["reviewerID"])["index"].max()
    minIdx = reviews.groupby(["reviewerID"])["index"].min()
    reviews["maxIdx"] = reviews["reviewerID"].map(maxIdx)
    reviews["minIdx"] = reviews["reviewerID"].map(minIdx)
    reviews["ordered"] = reviews["index"].map(int) - reviews["minIdx"].map(int) + 1
    reviews["count"] = reviews["maxIdx"].map(int) - reviews["minIdx"].map(int) + 1
    reviews["pct"] = 1.0 * reviews["ordered"].map(float) / reviews["count"].map(float)

    test_pct = float(args.ratio_test)
    validation_pct = float(args.ratio_test + args.ratio_validation)

    print("Getting list of train/validation/test instances")
    testIdx = list(reviews[(reviews["pct"] <= test_pct)]["id"])
    validationIdx = list(
        reviews[((reviews["pct"] > test_pct) & (reviews["pct"] <= validation_pct))]["id"]
    )
    trainIdx = list(reviews[(reviews["pct"] > validation_pct)]["id"])

    test = profile[(profile["id"].isin(testIdx))]
    validation = profile[(profile["id"].isin(validationIdx))]
    train = profile[(profile["id"].isin(trainIdx))]

    print("Filtering out unseen data in validation/test")
    users, items, aspects, opinions = (
        list(train["reviewerID"]),
        list(train["asin"]),
        list(train["aspect"]),
        list(train["opinion"]),
    )
    test = test[
        (
            test["reviewerID"].isin(users)
            & test["asin"].isin(items)
            & test["opinion"].isin(opinions)
            & test["aspect"].isin(aspects)
        )
    ]
    validation = validation[
        (
            validation["reviewerID"].isin(users)
            & validation["asin"].isin(items)
            & validation["opinion"].isin(opinions)
            & validation["aspect"].isin(aspects)
        )
    ]

    with open(os.path.join(args.out, SPLITTED_INFO_FILENAME), "w") as f:
        f.write("count_by,n_train,n_validation,n_test\n")
        f.write("%s,%d,%d,%d\n" % ("n_sentence", len(train), len(validation), len(test)))
        f.write(
            "%s,%d,%d,%d\n"
            % (
                "n_review",
                len(set(train["id"])),
                len(set(validation["id"])),
                len(set(test["id"])),
            )
        )
        f.write("\n")
        f.write("# aspect: %d\n" % len(set(aspects)))
        f.write("# opinion: %d\n" % len(set(opinions)))

    test.drop(columns=["id"]).to_csv(
        os.path.join(args.out, SPLITTED_TEST_FILE), index=False
    )
    validation.drop(columns=["id"]).to_csv(
        os.path.join(args.out, SPLITTED_VALIDATION_FILE), index=False
    )
    train_file = os.path.join(args.out, SPLITTED_TRAIN_FILE)
    train.drop(columns=["id"]).to_csv(train_file, index=False)

    # export dictionary: users, items, aspects, opinions
    save_dict(to_dict(users), os.path.join(args.out, USER_DICT_FILENAME))
    save_dict(to_dict(items), os.path.join(args.out, ITEM_DICT_FILENAME))
    save_dict(to_dict(aspects), os.path.join(args.out, ASPECT_DICT_FILENAME))
    save_dict(to_dict(opinions), os.path.join(args.out, OPINION_DICT_FILENAME))

    # export data for EFM/MTER
    print("Exporting data for EFM/MTER training")
    train[["reviewerID", "asin", "overall", "unixReviewTime"]].drop_duplicates().to_csv(
        os.path.join(args.out, "train.txt"), header=False, index=False
    )
    test[["reviewerID", "asin", "overall", "unixReviewTime"]].drop_duplicates().to_csv(
        os.path.join(args.out, "test.txt"), header=False, index=False
    )
    profile["aspect_sentiment"] = (
        profile["aspect"].map(str)
        + ":"
        + profile["opinion"].map(str)
        + ":"
        + profile["sentiment"].map(str)
    )
    sentiment = (
        profile.groupby(["reviewerID", "asin"])["aspect_sentiment"]
        .apply(list)
        .reset_index()
    )
    sentiment["aspect_sentiment"] = sentiment["aspect_sentiment"].apply(
        lambda x: ",".join(x)
    )
    with open(os.path.join(args.out, "sentiment.txt"), "w") as f:
        for row in sentiment.itertuples():
            f.write("{}\n".format(",".join([row[1], row[2], row[3]])))

    print("Done")

if __name__ == '__main__':
    main(parse_arguments())