# Hyperparameter Optimization
import optuna
from optuna_dashboard import run_server
import pickle
from mtg.Model_Evaluation import evaluate_model
from mtg.ml.generator import DraftGenerator, create_train_and_val_gens
from mtg.ml.models import DraftBot
from mtg.ml.trainer import Trainer
import tensorflow as tf
from optuna.integration.tensorboard import TensorBoardCallback
import os

def create_model(expansion_fname, batch_size,
                emb_dim, num_encoder_heads, num_decoder_heads, pointwise_ffn_width,
                num_encoder_layers, num_decoder_layers, emb_dropout, transformer_dropout, out_dropout,
                lr_warmup, emb_margin, emb_lambda):
    with open(expansion_fname, "rb") as f:
        expansion = pickle.load(f)
    
    # -----
    # to remove card information (scryfall) data, we only keep name and idx columns
    # comment out to keep card info and/or ratings
    expansion.cards = expansion.cards[["name", "idx"]]
    
    # to remove card ratings (17lands) data, we set every value to 0
    #for col in expansion.card_data_for_ML.columns:
    #    expansion.card_data_for_ML[col].values[:] = 0

    train_gen, val_gen = create_train_and_val_gens(
        expansion.draft,
        expansion.cards.copy(),
        train_p=1.0,
        id_col="draft_id",
        train_batch_size=batch_size,
        generator=DraftGenerator,
        include_val=True,
        external_val=True,
        part_train=True,
        part_val=False,
    )

    model = DraftBot(
        expansion=expansion,
        emb_dim=emb_dim,
        num_encoder_heads=num_encoder_heads,
        num_decoder_heads=num_decoder_heads,
        pointwise_ffn_width=pointwise_ffn_width,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        emb_dropout=emb_dropout,
        memory_dropout=transformer_dropout,
        out_dropout=out_dropout,
        name="DraftBot",
    )
    model.compile(
        learning_rate={"warmup_steps": lr_warmup},
        margin=emb_margin,
        emb_lambda=emb_lambda,
        #rare_lambda=rare_lambda,
        #cmc_lambda=cmc_lambda,
    )

    return model, train_gen, val_gen

def objective(trial):
    #n_train_iter = 10
    
    # Define the search space for hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    emb_dim = trial.suggest_categorical('emb_dim', [128, 256])
    num_encoder_heads = trial.suggest_categorical('num_encoder_heads', [16])
    num_decoder_heads = trial.suggest_categorical('num_decoder_heads', [16])
    pointwise_ffn_width = trial.suggest_categorical('pointwise_ffn_width', [256])
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [2])
    num_decoder_layers = trial.suggest_categorical('num_decoder_layers', [2])
    #emb_dropout = trial.suggest_float('emb_dropout', 0, 0.2, step=0.1)
    emb_dropout = trial.suggest_categorical('emb_dropout', [0.1])
    #transformer_dropout = trial.suggest_float('transformer_dropout', 0, 0.2, step=0.1)
    transformer_dropout = trial.suggest_categorical('transformer_dropout', [0.1])
    #out_dropout = trial.suggest_float('out_dropout', 0, 0.2, step=0.1)
    out_dropout = trial.suggest_categorical('out_dropout', [0])
    lr_warmup = trial.suggest_categorical('lr_warmup', [1000, 1500])
    #emb_margin = trial.suggest_float('emb_margin', 0.5, 1.5, step=0.5)
    emb_margin = trial.suggest_categorical('emb_margin', [1])
    #emb_lambda = trial.suggest_float('emb_lambda', 0.1, 0.5, step=0.2)
    emb_lambda = trial.suggest_categorical('emb_lambda', [0.5])
    epochs = trial.suggest_categorical('epochs', [1])

    # Create your custom Transformer model using the hyperparameters
    model, train_gen, val_gen = create_model("mtg/data/expansiontrain_full.pkl", batch_size, emb_dim, num_encoder_heads, num_decoder_heads, pointwise_ffn_width,
                        num_encoder_layers, num_decoder_layers, emb_dropout, transformer_dropout, out_dropout,
                        lr_warmup, emb_margin, emb_lambda)

    trainer = Trainer(
        model,
        generator=train_gen,
        val_generator=val_gen
        )
    trainer.train(
        epochs,
        batch_size=batch_size,
        print_keys=[#"prediction_loss", "embedding_loss"
                    #, "rare_loss", "cmc_loss"
                    ],
        verbose=True,
        )

    # # Using pruner:
    # for step in range(n_train_iter):
    #     trainer = Trainer(
    #     model,
    #     generator=train_gen,
    #     val_generator=val_gen
    #     )
    #     # Train the model
    #     last_valtop1 = trainer.train(
    #     epochs,
    #     batch_size=batch_size,
    #     print_keys=[#"prediction_loss", "embedding_loss"
    #                 #, "rare_loss", "cmc_loss"
    #                 ],
    #     verbose=True,
    #     pruning=True,
    #     train_batches=10
    # )
    #     intermediate_value = last_valtop1
    #     intermediate_value
    #     trial.report(intermediate_value, step)

    #     if trial.should_prune():
    #         raise optuna.TrialPruned()

    # Evaluate the model on the validation set and compute the objective value (accuracy)
    validation_accuracy, _, _ = evaluate_model(model, "mtg/data/expansiontrain_full.pkl", batch_size=batch_size, part_val=True)
    objective_value = validation_accuracy  # Maximizing val accuracy

    return objective_value

study_name="no_perf_batch_sizes"
if not os.path.exists("mtg/logs/"+study_name):
    os.mkdir("mtg/logs/"+study_name)
tensorboard_callback = TensorBoardCallback("mtg/logs/"+study_name+"/", metric_name="accuracy")
#storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(study_name=study_name,
                            direction='maximize', 
                            pruner=optuna.pruners.PatientPruner(optuna.pruners.SuccessiveHalvingPruner(reduction_factor=3, min_early_stopping_rate=2), patience=1),
                            #storage=storage
                            )
study.optimize(objective, n_trials=12, callbacks=[tensorboard_callback])

print(optuna.importance.get_param_importances(study))
fig = optuna.visualization.plot_param_importances(study)
#fig.show()
if not os.path.exists("mtg/images/"+study_name):
    os.mkdir("mtg/images/"+study_name)
fig.write_image("mtg/images/"+study_name+"/hp_importance.png")

#run_server(storage)

############# Early stopping?????
# Get the best hyperparameters and the corresponding objective value
#best_params = study.best_params
#best_value = study.best_value