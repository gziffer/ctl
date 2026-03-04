# Mitigating Spatio-Temporal Domain Shifts in EO Image Streams via Continuous Transfer Learning

## Setup & Installation

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ctl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Download Dataset
Download the DynamicEarthNet dataset (images resized to 224x224) from the provided S3 bucket and unzip it:
```bash
cd ctl
mkdir datasets
cd datasets
aws s3 cp s3://ctl-datasets/DynamicEarthNet.zip ./DynamicEarthNet.zip
unzip DynamicEarthNet.zip -d DynamicEarthNet
```

## Usage

### 1. Extract Embeddings

Extract embeddings from trained models:

```bash
python extract_embeddings.py \
  --extractor <multiutae or dinov3> \
  --csv_path <path_to_output_csv> \
  --batch_size <batch_size>  \
  --data_path <path_to_dataset> \
```

For DINOv3, you can change the pre-trained backbone when creating the model, default is `vit_large_patch16_dinov3.sat493m`. 

### 2. Preprocess Embeddings

Apply dimensionality reduction and normalization to embeddings:

```bash
python preprocessing.py \
  --input_csv <path_to_embeddings_csv> \
  --output_dir <output_directory> \
  --projection <projection_name> \
  --n_components <number_of_components> \
  --normalize <normalization_method>
```

**Available projections:**
- `pca`: Principal Component Analysis
- `srp`: Sparse Random Projection
- `grp`: Gaussian Random Projection

**Optional normalization:**
- `l1`: L1 norm
- `l2`: L2 norm

### 3. Run Streaming Experiments

#### Temporal Shift Experiment
Evaluate temporal domain shifts:

```bash
python run.py \
  --mode temporal \
```

#### Spatial Shift Experiment
Evaluate spatial domain shifts:

```bash
python run.py \
  --mode spatial \
```

**Flags:**
- `--adapt`: If added in the command, it enables prequential evaluation (optional)

## Configuration

Key configurations are set in `run.py`:
- `MONTHS_PER_YEAR`: Training window duration (default: 12)
- `RANDOM_SEED`: Reproducibility seed (default: 42)
- `MODELS`: Available streaming models (HoeffdingTree, SAMkNN, etc.)

Data splits for spatial domain shift experiments are defined in `split.json`.