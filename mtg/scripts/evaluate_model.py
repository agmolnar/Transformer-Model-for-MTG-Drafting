#evaluate_model
from mtg.ml.utils import load_model
from mtg.Model_Evaluation import evaluate_model, save_acc

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_full__weighted", "mtg/data/expansiontrain_full.pkl", batch_size=32, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_full__weighted", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_full_regularized", "mtg/data/expansiontrain_full.pkl", batch_size=32, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_full_regularized", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_base_best", "mtg/data/expansiontrain_full.pkl", batch_size=16, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_base_best", top1, top2, top3, rank_accs, pick_accs)

top1, top2, top3, rank_accs, pick_accs = evaluate_model("mtg/data/draft_model_noperf_best", "mtg/data/expansiontrain_full.pkl", batch_size=16, per_rank=True, per_pick=True, test_set=True)
save_acc("mtg/data/draft_model_noperf_best", top1, top2, top3, rank_accs, pick_accs)