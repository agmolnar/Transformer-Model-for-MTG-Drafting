import pickle
import os
import pandas as pd
from mtg.Model_Evaluation import evaluate_model
from mtg.Model_Evaluation import print_acc
pd.set_option('display.max_rows', 50)
df = pd.read_csv(r"mtg/data/finaltest.csv")
df.head()
os.getcwd()
#os.chdir(r"C:/Users/aguay/Documents/GitHub/rsaxe no modz/mtg")
df = pickle.load(open("mtg/data/expansiontrain_full.pkl", 'rb'))
df.get_mapping("idx", "name", include_basics=False)
df2 = pickle.load(open("mtg/data/expansiontest_fullfixed.pkl", 'rb'))
df2.get_mapping("idx", "name", include_basics=False)

column_order = df.draft.columns
df2.draft = df2.draft.reindex(columns=column_order)
df.draft.columns[11:]
df2.draft.columns[11:]

sum(df.cards["idx"] == df2.cards["idx"])
sum(df.cards["name"] == df2.cards["name"])
df.cards.columns
df.card_data_for_ML == df2.card_data_for_ML
df.cards.columns.sort_values() == df2.cards.columns.sort_values()
df2.cards.columns
df2.card_data_for_ML
df2.cards.columns.sort_values()
true_count = (df.card_data_for_ML == df2.card_data_for_ML).values.sum()
print("Number of True values:", true_count)
false_count = len(df.card_data_for_ML) - true_count
print("Number of False values:", false_count)
len(df2.card_data_for_ML)
import numpy as np
df.card_data_for_ML == df2.card_data_for_ML
df.card_data_for_ML != df2.card_data_for_ML
# Perform the comparison and get the indices of False values
false_indices = np.where(df.cards != df2.cards)

# Separate the row indices and column indices
rows, columns = false_indices
column_names = df.cards.columns[columns]

# Print the column names where False values are present
for column_name in column_names:
    print("False value in column", column_name)

###########
# Fixing the expansion files

# Load the first DataFrame from the .pkl file
with open("mtg/data/expansiontrain_full.pkl", "rb") as file:
    train_exp = pickle.load(file)

# Load the second DataFrame from the .pkl file
with open("mtg/data/expansionval_full.pkl", "rb") as file:
    val_exp = pickle.load(file)

# Load the third DataFrame from the .pkl file
with open("mtg/data/expansiontest_full.pkl", "rb") as file:
    test_exp = pickle.load(file)

column_order = train_exp.draft.columns
val_exp.draft = val_exp.draft.reindex(columns=column_order)
test_exp.draft = test_exp.draft.reindex(columns=column_order)

val_exp.cards = train_exp.cards
test_exp.cards = train_exp.cards

val_exp.card_data_for_ML = train_exp.card_data_for_ML
test_exp.card_data_for_ML = train_exp.card_data_for_ML

# Save the updated df2 to a new .pkl file
with open("mtg/data/expansionval_fullfixed.pkl", "wb") as file:
    pickle.dump(val_exp, file)

# Save the updated df2 to a new .pkl file
with open("mtg/data/expansiontest_fullfixed.pkl", "wb") as file:
    pickle.dump(test_exp, file)

type(df.cards)
# need name, idx, id
df.cards.info()
df.cards = df.cards[["name", "idx"]]
df.draft.tail()
df.draft.shape
n_rows_to_remove = 5779788
df.draft = df.draft.drop(df.draft.tail(n_rows_to_remove).index)
df.card_data_for_ML
for col in df.card_data_for_ML.columns:
    df.card_data_for_ML[col].values[:] = 0
df2 = df.card_data_for_ML.replace(df.card_data_for_ML, 0)
df.cards.copy().drop

attr = pickle.load(open(r"mtg/data/draft_model_base/attrs.pkl", 'rb'))
attr.keys()

from mtg.ml.display import draft_sim
from mtg.ml.utils import load_model
import pickle

# assume draft_model and build_model are pretrained instances of those MTG models
# assume expansion is a loaded instance of the expansion object containing the 
#     data corresponding to draft_model and build_model
# then, draft_sim as ran below will spin up a table of 8 bots and run them through a draft.
#       what is returned is links to 8 corresponding 17land draft logs and sealeddeck.tech deck builds.

with open("mtg/data/expansiontest_fullfixed.pkl", "rb") as f:
    expansion = pickle.load(f)

# column_order = expansion.draft.columns
# with open("mtg/data/expansionval_full.pkl", "rb") as f:
#     valexpansion = pickle.load(f)
# valexpansion.draft = valexpansion.draft.reindex(columns=column_order)

model, attrs = load_model(r"mtg/data/draft_model_full")

import tensorflow as tf
print(tf.version.VERSION)
token = "5e9200e8034842f0bb75ed0ad50dfc51" #replace this with your 17lands API token
bot_table = draft_sim(expansion, model, token=token, build_model=None)
bot_table

from mtg.ml.display import save_att_to_dir
from mtg.ml.display import draft_log_ai

redraft = draft_log_ai(
    "https://www.17lands.com/draft/65a6fe63104645ea9440e3bab68f565b",
    draft_model,
    expansion,
    batch_size=1,
    token="5e9200e8034842f0bb75ed0ad50dfc51",
    build_model=None,
    mod_lookup=dict(),
    basic_prior=True,
    att_folder=r"mtg/att_plots",
)
redraft

from mtg.ml.utils import load_model
from mtg.Model_Evaluation import evaluate_model, save_acc


top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_full", "mtg/data/expansiontrain_full.pkl", batch_size=32, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_full", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_full_best", "mtg/data/expansiontrain_full.pkl", batch_size=16, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_full_best", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_base", "mtg/data/expansiontrain_full.pkl", batch_size=32, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_base", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_base_best", "mtg/data/expansiontrain_full.pkl", batch_size=16, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_base_best", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_noperf", "mtg/data/expansiontrain_full.pkl", batch_size=32, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_noperf", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_noperf_best", "mtg/data/expansiontrain_full.pkl", batch_size=64, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_noperf_best", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_noperf_best2", "mtg/data/expansiontrain_full.pkl", batch_size=16, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_noperf_best2", top1, top2, top3, rank_accs, pick_accs)



import optuna
from optuna_dashboard import run_server

def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)

from mtg.ml.generator import DraftGenerator, create_train_and_val_gens
import pickle

with open("mtg/data/expansiontrain_full.pkl", "rb") as f:
        expansion = pickle.load(f)

data=expansion.draft
#expansion.draft["rank"] = expansion.draft["rank"].fillna("bronze")
#expansion.draft["rank"].value_counts()

train_gen, val_gen = create_train_and_val_gens(
    expansion.draft,
    expansion.cards.copy(),
    train_p=1.0,
    id_col="draft_id",
    train_batch_size=32,
    generator=DraftGenerator,
    include_val=True,
    weights=True,
    external_val=True,
    part_train=False,
    part_val=False,
    test_set=False,
    add_ranks=False,
    )

train_gen.weights
val_gen.weights
train_gen.draft_ids
train_gen.ranks
val_features, true_picks, weights, ranks = val_gen[0]
val_features
ranks


import pickle

with open("mtg/data/expansiontest_fullfixed.pkl", "rb") as f:
        expansion = pickle.load(f)

import pandas as pd

df=expansion.draft
df["rank"] = df["rank"].fillna("bronze")

#distributions of rank in original dataset

#platinum    2912155
#proportion  0.38726

#gold        1470430
#proportion  0.19554

#silver      1102957
#proportion  0.14667

#diamond     1032125
#proportion  0.13725

#mythic      502273
#proportion  0.06679

#bronze      499954
#proportion  0.06648

#total       7519894

d = {'platinum': 0.38726, 'gold': 0.19554, 'silver': 0.14667
    , 'diamond': 0.13725, 'mythic': 0.06679, 'bronze': 0.06648}

rankproportions = pd.Series(data=d, index=['platinum', 'gold', 'silver'
                    , 'diamond', 'mythic', 'bronze'])

dfrankproportions = df['rank'].value_counts()/sum(df['rank'].value_counts())
dfrankproportions-rankproportions

# Difference in proportions:

# Train:
# platinum    0.000934
# gold        0.000133
# silver     -0.000511
# diamond     0.000025
# mythic     -0.000024
# bronze     -0.000546

# Val:
# platinum    0.001460
# gold        0.001076
# silver     -0.000590
# diamond    -0.002225
# mythic      0.003317
# bronze     -0.003028

# Test:
# platinum   -0.004632
# gold        0.000061
# silver      0.004599
# diamond     0.003698
# mythic     -0.001703
# bronze     -0.002013