# SEER

This is the code for the paper:

**[Synthesizing Aspect-Driven Recommendation Explanations from Reviews](https://lthoang.com/assets/publications/ijcai20.pdf)**
<br>
[Trung-Hoang Le](http://lthoang.com/) and [Hady W. Lauw](http://www.hadylauw.com/)
<br>
Presented at [IJCAI-PRICAI-20](https://ijcai20.org/)

We provide:
- Code to run SEER framework
- [Data](https://github.com/PreferredAI/seer/tree/master/data) we used for our experiments


If you find the code and data useful in your research, please cite:

```
@inproceedings{ijcai2020-336,
  title     = {Synthesizing Aspect-Driven Recommendation Explanations from Reviews},
  author    = {Le, Trung-Hoang and Lauw, Hady W.},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {2427--2434},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/336},
  url       = {https://doi.org/10.24963/ijcai.2020/336},
}
```

## How to run

```
pip install -r requirements.txt
```

The following execution scripts are constructed for `toy` category as in the paper.

### Prepare data
```
python prepare_data.py --input data/toy/profile.csv --out data/toy --ratio_validation 0.2 --ratio_test 0.2
```

### Train aspect-level sentiments model (EFM or MTER)

```
python train_efm.py --indir data/toy --epoch 1000 --out data/toy/efm
```

```
python train_mter.py --indir data/toy --epoch 100000 --out data/toy/mter
```

### Train Aspect-Sentiment Context2Vec (ASC2V) model for opinion completion task

Execute the following command to prepare training data for this task
```
python prepare_opinion_contextualization_data.py --indir data/toy --efm_dir data/toy/efm --mter_dir data/toy/mter --out data/toy
```

Train ASC2V by the following command:
```
python train_asc2v.py --indir data/toy --gpu 0 --out data/toy/asc2v --context asc2v
```
For ASC2V with sentiment score from MTER model, executing the following command:
```
python train_asc2v.py --indir data/toy --gpu 0 --out data/toy/asc2v-mter --context asc2v-mter
```

### Synthesizing explanation

```
python seer.py --input A10L9NQO44OLOU,B0044T2KBU,toy,toy --corpus_path data/toy/train.csv --strategy greedy-efm --preference_dir data/toy/efm --contextualizer_path data/toy/asc2v/model.params
```
or if you want to use the sentiments produced by MTER
```
python seer.py --input A10L9NQO44OLOU,B0044T2KBU,toy,toy --corpus_path data/toy/train.csv --strategy greedy-mter --preference_dir data/toy/mter --contextualizer_path data/toy/asc2v-mter/model.params
```

The above command generates an explanation for user `A10L9NQO44OLOU`, item `B0044T2KBU`, and the demanded aspects are `toy,toy` (2 sentences about aspect `toy`). This example is taken from the test dataset.

By default, we run SEER with greedy algorithm. SEER-ILP is provided and [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/sg-en/products/ilog-cplex-optimization-studio) must be ready to be able to proceed further. After setting up CPLEX, you can try SEER-ILP with argument `--strategy ilp-efm` (or `--strategy ilp-mter`).

To run the framework with other data, please modify the arguments accordingly.

## Contact
Questions and discussion are welcome: [lthoang.com](http://lthoang.com)

