import argparse
import os

import cornac
import numpy as np
import pandas as pd
from cornac.data import Reader, SentimentModality
from cornac.eval_methods import BaseMethod


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--indir", default="data/toy", help="Input data directory"
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=100000, help="Max number of iterations"
    )
    parser.add_argument(
        "-o", "--out", default="data/toy/mter", help="Directory to output the result"
    )
    parser.add_argument("-uf", "--user_factors", type=int, default=15)
    parser.add_argument("-if", "--item_factors", type=int, default=15)
    parser.add_argument("-af", "--aspect_factors", type=int, default=12)
    parser.add_argument("-of", "--opinion_factors", type=int, default=12)
    parser.add_argument("-bs", "--bpr_samples", type=int, default=1000)
    parser.add_argument("-es", "--element_samples", type=int, default=50)
    parser.add_argument("-reg", "--lambda_reg", type=float, default=0.1)
    parser.add_argument("-bpr", "--lambda_bpr", type=float, default=10.0)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument(
        "-rs",
        "--seed",
        type=int,
        default=None,
        help="Random Seed Value",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    print("Input directory:", args.indir)
    print("Output directory:", args.out)
    print("# epoch:", args.epoch)
    print("# user factors:", args.user_factors)
    print("# item factors:", args.item_factors)
    print("# aspect factors:", args.aspect_factors)
    print("# opinion factors:", args.opinion_factors)
    print("# bpr samples:", args.bpr_samples)
    print("# element samples:", args.element_samples)
    print("lambda reg =", args.lambda_reg)
    print("lambda bpr =", args.lambda_bpr)
    print("learning rate =", args.learning_rate)
    print("Seed value =", args.seed)
    print("VERBOSE =", args.verbose)
    return args


args = parse_arguments()

os.makedirs(args.out, exist_ok=True)

reader = Reader()
train_data = reader.read(os.path.join(args.indir, "train.txt"), sep=",")
test_data = reader.read(os.path.join(args.indir, "test.txt"), sep=",")
sentiment = reader.read(
    os.path.join(args.indir, "sentiment.txt"), fmt="UITup", sep=",", tup_sep=":"
)
md = SentimentModality(data=sentiment)
eval_method = BaseMethod.from_splits(
    train_data=train_data,
    test_data=test_data,
    sentiment=md,
    exclude_unknowns=True,
    verbose=args.verbose,
)

mter = cornac.models.MTER(
    n_user_factors=args.user_factors,
    n_item_factors=args.item_factors,
    n_aspect_factors=args.aspect_factors,
    n_opinion_factors=args.opinion_factors,
    n_bpr_samples=args.bpr_samples,
    n_element_samples=args.element_samples,
    lambda_reg=args.lambda_reg,
    lambda_bpr=args.lambda_bpr,
    max_iter=args.epoch,
    lr=args.learning_rate,
    verbose=args.verbose,
    seed=args.seed,
)


exp = cornac.Experiment(
    eval_method=eval_method,
    models=[mter],
    metrics=[
        cornac.metrics.RMSE(),
        cornac.metrics.Recall(k=10),
        cornac.metrics.Recall(k=50),
        cornac.metrics.NDCG(k=50),
        cornac.metrics.AUC(),
    ],
)

exp.run()

# save params and trained weights
pd.DataFrame(
    data={
        "raw_id": list(eval_method.train_set.uid_map.keys()),
        "id": list(eval_method.train_set.uid_map.values()),
    }
)[["raw_id", "id"]].to_csv(os.path.join(args.out, "uid_map"), header=None, index=None)
pd.DataFrame(
    data={
        "raw_id": list(eval_method.train_set.iid_map.keys()),
        "id": list(eval_method.train_set.iid_map.values()),
    }
)[["raw_id", "id"]].to_csv(os.path.join(args.out, "iid_map"), header=None, index=None)
pd.DataFrame(
    data={
        "raw_id": list(eval_method.sentiment.aspect_id_map.keys()),
        "id": list(eval_method.sentiment.aspect_id_map.values()),
    }
)[["raw_id", "id"]].to_csv(
    os.path.join(args.out, "aspect_id_map"), header=None, index=None
)
pd.DataFrame(
    data={
        "raw_id": list(eval_method.sentiment.opinion_id_map.keys()),
        "id": list(eval_method.sentiment.opinion_id_map.values()),
    }
)[["raw_id", "id"]].to_csv(
    os.path.join(args.out, "opinion_id_map"), header=None, index=None
)

np.save(os.path.join(args.out, "U"), mter.U)
np.save(os.path.join(args.out, "I"), mter.I)
np.save(os.path.join(args.out, "A"), mter.A)
np.save(os.path.join(args.out, "O"), mter.O)
np.save(os.path.join(args.out, "G1"), mter.G1)
np.save(os.path.join(args.out, "G2"), mter.G2)
np.save(os.path.join(args.out, "G3"), mter.G3)
