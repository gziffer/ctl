import os
import argparse
import pandas as pd
import numpy as np

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer


# ---------------- Projection registry ----------------
PROJECTIONS = {
    "srp": SparseRandomProjection,
    "grp": GaussianRandomProjection,
    "pca": PCA,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply dimensionality reduction to embeddings."
    )

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--projection",
        type=str,
        choices=["srp", "grp", "pca"],
        required=True
    )

    parser.add_argument("--n_components", type=int, required=True)

    parser.add_argument(
        "--normalize",
        type=str,
        choices=["l1", "l2"],
        default=None,
        help="Optional normalization after projection"
    )

    parser.add_argument("--emb_prefix", type=str, default="emb_")
    parser.add_argument("--random_state", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    print("\nLoading CSV...")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded dataset: {df.shape}")

    # ---------------- Extract embeddings ----------------
    emb_cols = [c for c in df.columns if c.startswith(args.emb_prefix)]
    other_cols = [c for c in df.columns if c not in emb_cols]

    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found. Check emb_prefix.")

    X = df[emb_cols].values.astype(np.float32)
    print(f"Embedding matrix shape: {X.shape}")

    # ---------------- Projection ----------------
    proj_class = PROJECTIONS[args.projection]

    print(f"\nApplying {args.projection.upper()} → {args.n_components} dims")

    if args.projection == "pca":
        model = proj_class(
            n_components=args.n_components,
            random_state=args.random_state,
            svd_solver="randomized"
        )
    else:
        model = proj_class(
            n_components=args.n_components,
            random_state=args.random_state
        )

    X_proj = model.fit_transform(X)
    print(f"Projected shape: {X_proj.shape}")

    # ---------------- Optional Normalization ----------------
    if args.normalize is not None:
        print(f"Applying {args.normalize.upper()} normalization")
        normalizer = Normalizer(norm=args.normalize)
        X_proj = normalizer.fit_transform(X_proj)

    # ---------------- Build dataframe ----------------
    df_proj = pd.DataFrame(
        X_proj.astype("float32"),
        columns=[f"emb_{i+1}" for i in range(args.n_components)]
    )

    df_final = pd.concat(
        [df[other_cols].reset_index(drop=True), df_proj],
        axis=1
    )

    # ---------------- Save ----------------
    os.makedirs(args.output_dir, exist_ok=True)

    input_name = os.path.splitext(os.path.basename(args.input_csv))[0]

    norm_tag = f"_{args.normalize}" if args.normalize else ""
    output_path = os.path.join(
        args.output_dir,
        f"{input_name}_{args.projection}{args.n_components}{norm_tag}.csv"
    )

    df_final.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Final shape: {df_final.shape}")


if __name__ == "__main__":
    main()