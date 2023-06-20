# Model Evaluation
from mtg.ml.generator import DraftGenerator, create_train_and_val_gens
from mtg.ml.models import DraftBot
import pickle
from mtg.ml.trainer import Trainer
import tensorflow as tf
import numpy as np
from mtg.ml.utils import load_model
import csv

def evaluate_model(model_path, path_expansion_train, batch_size = 32, per_pick=False, test_set=False, part_val=False, per_rank=False):
    with open(path_expansion_train, "rb") as f:
        expansion = pickle.load(f)
    
    if isinstance(model_path, str):
        model, attrs = load_model(model_path)
    else:
        model = model_path
    
    _, val_gen = create_train_and_val_gens(
            expansion.draft,
            expansion.cards.copy(),
            id_col="draft_id",
            train_batch_size=batch_size,
            generator=DraftGenerator,
            include_val=True,
            external_val=True,
            part_train=False,
            part_val=part_val,
            test_set=test_set,
            add_ranks=per_rank
        )
    n_batches = len(val_gen)
    top1_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    top2_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)
    top3_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    bronze_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    silver_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    gold_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    platinum_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    diamond_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    mythic_accuracy = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    pick_accuracy = [tf.keras.metrics.Accuracy() for _ in range(42)]
    for i in range(n_batches):
            val_features, true_picks, _, ranks = val_gen[i]
            val_output = model(val_features, training=False)
            val_output = val_output[0]
            for d in range(len(val_output)):
                pred = val_output[d]
                true = true_picks[d]
                rank = ranks[d]
                update_result1 = top1_accuracy.update_state(true, pred)
                update_result2 = top2_accuracy.update_state(true, pred)
                update_result3 = top3_accuracy.update_state(true, pred)
                _ = tf.identity(update_result1)
                _ = tf.identity(update_result2)
                _ = tf.identity(update_result3)
                if per_rank:
                    if rank[0] == 'bronze':
                        update_result_bronze = bronze_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_bronze)
                    elif rank[0] == 'silver':
                        update_result_silver = silver_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_silver)
                    elif rank[0] == 'gold':
                        update_result_gold = gold_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_gold)
                    elif rank[0] == 'platinum':
                        update_result_platinum = platinum_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_platinum)
                    elif rank[0] == 'diamond':
                        update_result_diamond = diamond_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_diamond)
                    elif rank[0] == 'mythic':
                        update_result_mythic = mythic_accuracy.update_state(true, pred)
                        _ = tf.identity(update_result_mythic)
                if per_pick:
                    update_results=[]
                    for p in range(42):
                        predictions_top1 = tf.argmax(pred, axis=-1).numpy()
                        pick_true = true[p].numpy()
                        pick_pred = predictions_top1[p]
                        update_results.append(pick_accuracy[p].update_state(pick_true, pick_pred))
                        _ = tf.identity(update_results[p])
    top1_acc = top1_accuracy.result().numpy()
    top2_acc = top2_accuracy.result().numpy()
    top3_acc = top3_accuracy.result().numpy()
    bronze_acc = bronze_accuracy.result().numpy()
    silver_acc = silver_accuracy.result().numpy()
    gold_acc = gold_accuracy.result().numpy()
    platinum_acc = platinum_accuracy.result().numpy()
    diamond_acc = diamond_accuracy.result().numpy()
    mythic_acc = mythic_accuracy.result().numpy()
    rank_accs = bronze_acc, silver_acc, gold_acc, platinum_acc, diamond_acc, mythic_acc
    pick_accs = [pick_metric.result().numpy() for pick_metric in pick_accuracy]
    if per_rank and per_pick:
        return top1_acc, top2_acc, top3_acc, rank_accs, pick_accs
    elif per_rank:
        return top1_acc, top2_acc, top3_acc, rank_accs
    elif per_pick:
        return top1_acc, top2_acc, top3_acc, pick_accs
    else:
        return top1_acc, top2_acc, top3_acc


def save_acc(model_path, top1_acc, top2_acc, top3_acc, rank_accs=None, pick_accs=None):
    # Specify the CSV file path and name
    csv_file = "/test_results.csv"
    # Define the data to write to the CSV file
    data = [
        ["Metric", "Accuracy"],
        ["Top-1 Accuracy", top1_acc],
        ["Top-2 Accuracy", top2_acc],
        ["Top-3 Accuracy", top3_acc]
    ]
    # Check if rank accuracies exist
    if rank_accs is not None:
        data.append(["Bronze Accuracy", rank_accs[0]])
        data.append(["Silver Accuracy", rank_accs[1]])
        data.append(["Gold Accuracy", rank_accs[2]])
        data.append(["Platinum Accuracy", rank_accs[3]])
        data.append(["Diamond Accuracy", rank_accs[4]])
        data.append(["Mythic Accuracy", rank_accs[5]])
    # Check if pick accuracies exist
    if pick_accs is not None:
        for pick, acc in enumerate(pick_accs, start=1):
            pack_num = (pick - 1) // 14 + 1  # Calculate the pack number
            card_num = (pick - 1) % 14 + 1  # Calculate the card number within the pack
            data.append(["Pack {} Pick {}".format(pack_num, card_num), acc])
    # Write the data to the CSV file
    with open(model_path+csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)
    # Print a message indicating the file has been written
    print("Accuracy results have been written to accuracy_results.csv.")

# print("Top-1 Accuracy:", top1_acc)
# print("Top-2 Accuracy:", top2_acc)
# print("Top-3 Accuracy:", top3_acc)
# if pick_accs is not None:
#     for pick, acc in enumerate(pick_accs, start=1):
#         pack_num = (pick - 1) // 14 + 1  # Calculate the pack number
#         card_num = (pick - 1) % 14 + 1  # Calculate the card number within the pack
#         print("Top-1 Accuracy for Pack", pack_num, "Pick", card_num, ":", acc)
#     # above result is not considering weighting
#     # base accuracy = 0.23225445189730903