import argparse
import os
import json

import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm

from capymoa.classifier import SAMkNN, HoeffdingTree
from capymoa.evaluation import ClassificationEvaluator
from capymoa.instance import LabeledInstance
from capymoa.stream import Schema

from metrics import StreamingChangeEvaluator, NUM_CLASSES, CLASS_NAMES


# ==========================================================
# -------------------- ARGPARSE ----------------------------
# ==========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True,
                    choices=["temporal", "spatial"],
                    help="Run temporal or spatial experiment")
parser.add_argument("--adapt", action="store_true",
                    help="Enable adaptation during streaming")
args = parser.parse_args()

MODE = args.mode
ADAPT_ON_STREAM = args.adapt


# ==========================================================
# -------------------- CONFIG ------------------------------
# ==========================================================

wandb.login()

PROJECT_NAME = (
    "capymoa-streaming-temporal" if MODE == "temporal" else
    "capymoa-streaming-spatial"
)

SPLITS_JSON_PATH = "split.json"

PATCH_ID = "patch_id"
LABEL = "label"
OTHER_FEATURES = ["sits_id", "timestamp"]

MONTHS_PER_YEAR = 12

RANDOM_SEED = 42

MODELS = {
    "HoeffdingTree": {
        "class": HoeffdingTree,
        "params": {
            "random_seed": RANDOM_SEED
        }
    }
}

# ==========================================================
# -------------------- TEMPORAL ----------------------------
# ==========================================================

def run_temporal_experiment(csv_path):

    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_ts = sorted(df["timestamp"].unique())

    train_ts = unique_ts[:MONTHS_PER_YEAR]
    stream_ts = unique_ts[MONTHS_PER_YEAR:]

    df_train = df[df["timestamp"].isin(train_ts)]
    df_stream = df[df["timestamp"].isin(stream_ts)]

    return run_core_loop(
        df_train,
        df_stream,
        csv_path,
        scenario_name="temporal"
    )


# ==========================================================
# -------------------- SPATIAL -----------------------------
# ==========================================================

def run_spatial_experiment(csv_path):

    with open(SPLITS_JSON_PATH) as f:
        splits = json.load(f)

    scenario = "spatial"

    df = pd.read_csv(csv_path)

    train_ids = splits[scenario]['train_ids']
    prequential_ids = splits[scenario]['prequential_ids']

    df_train = df[df['sits_id'].isin(train_ids)] \
        .sort_values("timestamp").reset_index(drop=True)

    df_stream = df[(df['timestamp'] >= 365) & (df['sits_id'].isin(prequential_ids))] \
        .sort_values("timestamp").reset_index(drop=True)

    return run_core_loop(
        df_train,
        df_stream,
        csv_path,
        scenario_name="spatial"
    )


# ==========================================================
# -------------------- CORE LOOP ---------------------------
# ==========================================================

def run_core_loop(df_train, df_stream, csv_path, scenario_name):

    if df_train.empty or df_stream.empty:
        print("Empty split. Skipping.")
        return []

    feature_cols = [
        c for c in df_train.columns
        if c not in [PATCH_ID, LABEL, *OTHER_FEATURES]
    ]

    schema = Schema.from_custom(
        feature_names=feature_cols,
        target_attribute_name=LABEL,
        values_for_class_label=list(range(len(CLASS_NAMES)))
    )

    stream_ts = sorted(df_stream["timestamp"].unique())
    run_name_base = os.path.basename(csv_path).replace(".csv", "")

    results = []

    for model_name, model_cfg in tqdm(MODELS.items(), desc=f"Models ({run_name_base})", leave=False):

        run_name = f"{run_name_base}_{scenario_name}_{model_name}"
        run = wandb.init(
            project=PROJECT_NAME,
            name=run_name,
            reinit=True
        )

        try:
            model = model_cfg["class"](schema=schema, **model_cfg["params"])

            std_eval_cum = ClassificationEvaluator(schema=schema)
            change_eval_cum = StreamingChangeEvaluator(num_classes=NUM_CLASSES)

            # -------- INITIAL TRAIN --------
            print(f"\nInitial training for {model_name}...")
            for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc=f"Initial Train ({model_name})", leave=False):
                y = int(row[LABEL])
                X = np.array([row[c] for c in feature_cols], dtype=float)
                instance = LabeledInstance.from_array(schema, x=X, y_index=y)
                model.train(instance)

            # -------- PREQUENTIAL --------
            progress = tqdm(stream_ts, desc=f"Prequential ({model_name})", leave=False)
            for ts in progress:

                df_month = df_stream[df_stream["timestamp"] == ts]
                std_eval_month = ClassificationEvaluator(schema=schema)
                
                for _, row in df_month.iterrows():

                    y_true = int(row[LABEL])
                    X = np.array([row[c] for c in feature_cols], dtype=float)
                    instance = LabeledInstance.from_array(schema, x=X, y_index=y_true)

                    y_pred = int(model.predict(instance))

                    std_eval_cum.update(y_true, y_pred)
                    std_eval_month.update(y_true, y_pred)
                    change_eval_cum.update(
                        row[PATCH_ID],
                        y_true,
                        y_pred
                    )
                    
                # ------- ADAPTATION (OPTIONAL) --------
                print("Adapting model on current month data..." if ADAPT_ON_STREAM else "Skipping adaptation...", end="\n\n")
                if ADAPT_ON_STREAM:
                    for _, row in df_month.iterrows():
                        y = int(row[LABEL])
                        X = np.array([row[c] for c in feature_cols], dtype=float)
                        instance = LabeledInstance.from_array(schema, x=X, y_index=y)
                        model.train(instance)
                
                                # Get cumulative metrics
                metrics_cum = change_eval_cum.compute(prefix="")
                log_cum_data = {**metrics_cum,
                            "accuracy": std_eval_cum.accuracy(),
                            "precision": std_eval_cum.precision(),
                            "recall": std_eval_cum.recall(),
                            "f1": std_eval_cum.f1_score(),
                           }
                
                # Get monthly metrics
                log_month_data = {
                            "month/accuracy": std_eval_month.accuracy(),
                            "month/precision": std_eval_month.precision(),
                            "month/recall": std_eval_month.recall(),
                            "month/f1": std_eval_month.f1_score(),
                            }
                
                wandb.log(log_cum_data, step=ts)
                wandb.log(log_month_data, step=ts)

            final_metrics = {
                "embedding": run_name_base,
                "model": model_name,
                "accuracy": std_eval_cum.accuracy(),
                "precision": std_eval_cum.precision(),
                "recall": std_eval_cum.recall(),
                "f1": std_eval_cum.f1_score(),
                **change_eval_cum.compute(),
            }

            results.append(final_metrics)

        finally:
            run.finish()

    return results


# ==========================================================
# -------------------- MAIN -------------------------------
# ==========================================================

if __name__ == "__main__":

    OUTPUT_CSV_FILE = (
        "final_temporal_results.csv"
        if MODE == "temporal"
        else "final_spatial_results.csv"
    )

    # Data path - UPDATE THIS TO YOUR LOCAL PATH
    file_path = "dinov3_embeddings.csv" 

    if MODE == "temporal":
        res = run_temporal_experiment(file_path)
    else:
        res = run_spatial_experiment(file_path)

    if res:
        df_batch = pd.DataFrame(res)
        write_header = not os.path.exists(OUTPUT_CSV_FILE)
        df_batch.to_csv(
            OUTPUT_CSV_FILE,
            mode='a',
            header=write_header,
            index=False
        )

    print(f"\nResults saved to {OUTPUT_CSV_FILE}")