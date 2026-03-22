<div align="center">
<h2>
Mitigating Spatio-Temporal Domain Shifts in Earth Observation Image Streams via Continuous Transfer Learning

<a >Giacomo Ziffer</a>&emsp;
<a >Lorenzo Iovine</a>&emsp;
<a >Chiara Thien Thao Nguyen Ba</a>&emsp;
<a >Emanuele Della Valle</a>

<p></p>

</h2>
</div>

This is the official implementation of **Mitigating Spatio-Temporal Domain Shifts in Earth Observation Image Streams via Continuous Transfer Learning**, submitted to **Neurocomputing**.

Satellite image streams are subject to continuous distribution shifts caused by seasonal changes, sensor variability, and geographic diversity. This work tackles these challenges by proposing a continuous transfer learning framework that adapts to spatio-temporal domain shifts in Earth Observation (EO) image streams. 

Our approach builds on the domain-shift analysis presented in [1], which studied semantic land-cover classification and change detection in satellite image time series. We evaluate our method on the **DynamicEarthNet** dataset [2], a large-scale, globally distributed, and multi-year satellite dataset designed for semantic change segmentation.

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
Download the DynamicEarthNet dataset (images resized to 224x224). You can download the dataset using the code below or by following this [link](https://drive.google.com/file/d/1MGJvAcNr0vTrcI2EDnP24vTB1A3NKTmx/view?usp=drive_link). 
```bash
cd ctl
mkdir datasets
cd datasets
gdown 1MGJvAcNr0vTrcI2EDnP24vTB1A3NKTmx
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

## References

[1] Elliot Vincent, Jean Ponce, and Mathieu Aubry. Satellite Image Time Series Semantic Change Detection: Novel Architecture and Analysis of Domain Shift. *arXiv*, 2024.

[2] Aysim Toker et al. DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 21158–21167, 2022.