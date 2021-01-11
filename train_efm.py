import pandas as pd
import numpy as np
import os
import argparse
import cornac
from cornac.data import SentimentModality, Reader
from cornac.eval_methods import BaseMethod


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--indir", default="data/toy", help="Input data directory"
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=1000, help="Max number of iterations"
    )
    parser.add_argument(
        "-o", "--out", default="data/toy/efm", help="Directory to output the result"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.85,
        help="Balance factor for EFM ranking score",
    )
    parser.add_argument("-ef", "--num_explicit_factors", type=int, default=40)
    parser.add_argument("-lf", "--num_latent_factors", type=int, default=60)
    parser.add_argument("-ca", "--num_most_cared_aspects", type=int, default=15)
    parser.add_argument(
        "-rs",
        "--seed",
        type=int,
        default=None,
        help="Random Seed Value",
    )
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    print("Input directory:", args.indir)
    print("Output directory:", args.out)
    print("# epoch:", args.epoch)
    print("alpha:", args.alpha)
    print("# explicit factors:", args.num_explicit_factors)
    print("# latent factors:", args.num_latent_factors)
    print("# most cared aspects:", args.num_most_cared_aspects)
    print("Seed value:", args.seed)
    print("")
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

efm = cornac.models.EFM(
    num_explicit_factors=args.num_explicit_factors,
    num_latent_factors=args.num_latent_factors,
    num_most_cared_aspects=args.num_most_cared_aspects,
    rating_scale=5.0,
    alpha=args.alpha,
    lambda_x=1,
    lambda_y=1,
    lambda_u=0.01,
    lambda_h=0.01,
    lambda_v=0.01,
    max_iter=args.epoch,
    trainable=True,
    verbose=args.verbose,
    seed=args.seed,
)

exp = cornac.Experiment(
    eval_method=eval_method,
    models=[efm],
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

np.save(os.path.join(args.out, "U1"), efm.U1)
np.save(os.path.join(args.out, "U2"), efm.U2)
np.save(os.path.join(args.out, "V"), efm.V)
np.save(os.path.join(args.out, "H1"), efm.H1)
np.save(os.path.join(args.out, "H2"), efm.H2)
