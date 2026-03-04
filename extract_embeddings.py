import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data import DynamicEarthNet
from models.networks.multiutae import MultiUTAE
import timm


# ==========================================================
# ---------------- BASE INTERFACE --------------------------
# ==========================================================

class BaseTemporalExtractor(nn.Module):
    """
    All extractors must return:
        embeddings: [B, T, P, C]
        labels:     [B, T, P]
        sits_id:    [B]
        positions:  [B, T]
    """

    def forward(self, batch):
        raise NotImplementedError


# ==========================================================
# ---------------- MULTIUTAE -------------------------------
# ==========================================================

class MultiUTAETemporalExtractor(BaseTemporalExtractor):

    def __init__(self, model, pool="avg", num_classes=6):
        super().__init__()
        self.model = model
        self.pool = pool
        self.num_classes = num_classes

    def forward(self, batch):

        x = batch["data"].float()          # [B,T,C,H,W]
        gt = batch["gt"]                  # [B,T,H,W]
        positions = batch["positions"].long()

        pad_mask = ((x == self.model.pad_value)
                    .all(dim=-1).all(dim=-1).all(dim=-1))

        out = self.model.in_conv.smart_forward(x)
        feature_maps = [out]

        for i in range(self.model.n_stages - 1):
            out = self.model.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        feat_last = feature_maps[-1]

        out_temporal, _ = self.model.temporal_encoder(
            feat_last,
            batch_positions=positions,
            pad_mask=pad_mask
        )

        if self.pool == "avg":
            emb = out_temporal.mean(dim=[-2, -1])  # [B,T,C]
        else:
            emb = out_temporal.amax(dim=[-2, -1])

        # Convert to patch dimension P=1
        emb = emb.unsqueeze(2)  # [B,T,1,C]

        labels = self.compute_majority(gt)  # [B,T]
        labels = labels.unsqueeze(2)       # [B,T,1]

        return {
            "embeddings": emb,
            "labels": labels,
            "sits_id": batch["sits_id"],
            "positions": positions,
        }

    def compute_majority(self, gt):
        B, T, H, W = gt.shape
        gt_flat = gt.view(B, T, -1)
        labels = torch.zeros((B, T), dtype=torch.long, device=gt.device)

        for b in range(B):
            for t in range(T):
                bincount = torch.bincount(
                    gt_flat[b, t],
                    minlength=self.num_classes
                )
                labels[b, t] = bincount.argmax()
        return labels


# ==========================================================
# ---------------- DINOv3 ----------------------------------
# ==========================================================

class DINOv3TemporalExtractor(BaseTemporalExtractor):

    def __init__(self, model, num_classes=6, input_channels=[0,1,2]):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.embedding_dim = model.num_features
        self.patch_size = model.patch_embed.patch_size[0]

    def forward(self, batch):

        x = batch["data"].float()  # [B,T,C,H,W]
        gt = batch["gt"]
        positions = batch["positions"].long()

        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        x = x[:, self.input_channels]

        tokens = self.model.forward_features(x)

        num_patches = (H // self.patch_size) * (W // self.patch_size)
        patch_tokens = tokens[:, -num_patches:, :]

        emb = patch_tokens.view(B, T, num_patches, -1)

        labels = self.compute_patch_majority(gt)

        return {
            "embeddings": emb,
            "labels": labels,
            "sits_id": batch["sits_id"],
            "positions": positions,
        }

    def compute_patch_majority(self, gt):

        B, T, H, W = gt.shape
        gt = gt.reshape(B*T, 1, H, W).float()

        patches = F.unfold(
            gt,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        labels, _ = torch.mode(patches, dim=1)
        num_patches = patches.shape[-1]

        return labels.view(B, T, num_patches).long()


# ==========================================================
# ---------------- CSV SAVER -------------------------------
# ==========================================================

def save_embeddings(batch_out, csv_path, mode, global_patch_id):

    emb = batch_out["embeddings"].detach().cpu().numpy()
    labels = batch_out["labels"].cpu().numpy()
    sits_id = batch_out["sits_id"].cpu().numpy()
    positions = batch_out["positions"].cpu().numpy()

    B, T, P, C = emb.shape
    rows = []

    for i in range(B):

        for p in range(P):
            patch_id = global_patch_id
            global_patch_id += 1

            for t in range(T):

                if emb[i, t, p].sum() == 0:
                    continue

                row = {
                    "sits_id": sits_id[i],
                    "patch_id": patch_id,
                    "timestamp": int(positions[i, t]),
                    "label": int(labels[i, t, p]),
                }

                row.update({
                    f"emb_{k}": float(emb[i, t, p, k])
                    for k in range(C)
                })

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, mode=mode, header=(mode=="w"), index=False)

    return global_patch_id


# ==========================================================
# ---------------- MAIN ------------------------------------
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor",
                        choices=["multiutae", "dinov3"],
                        required=True)
    parser.add_argument("--csv_path", type=str, required=True) # Path to save CSV embeddings
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_path", type=str, required=True) # Path to DynamicEarthNet dataset
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = DynamicEarthNet(
        path=args.data_path,
        split="train",
        domain_shift_type="temporal",
        train_length=24,
        img_size=224,
        date_aug_range=0
    )

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

    if args.extractor == "multiutae":

        model = MultiUTAE(
            input_dim=4,
            num_classes=6,
            in_features=512
        )

        extractor = MultiUTAETemporalExtractor(model)

    else:

        model = timm.create_model(
            "vit_large_patch16_dinov3.sat493m", # Pre-trained model selection
            pretrained=True,
            num_classes=6
        )

        extractor = DINOv3TemporalExtractor(model)

    extractor.to(device)
    extractor.eval()

    mode = "w"
    global_patch_id = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            out = extractor(batch)

            global_patch_id = save_embeddings(
                out,
                args.csv_path,
                mode,
                global_patch_id
            )

            mode = "a"

    df = pd.read_csv(args.csv_path)
    df.sort_values(by=["timestamp", "sits_id", "patch_id"], inplace=True)
    df.to_csv(args.csv_path, index=False)

    print("Done. Total rows:", len(df))