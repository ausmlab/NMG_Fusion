# NMG_Fusion
Point Cloud Multi Tasks(3D Semantics Segmentation and 2D Oriented Object detection)

 
# Installation (on cuda 11.x)

I used the below environment for Maryam’s fusion thing (working on both PTv3 and DINO based OBB)

### Create a conda virtual environment with Python 3.8 (Same with PTv3)
```
 Create a conda virtual environment with Python 3.8 
conda create -n fusion python=3.8 -y
conda activate fusion

# Install libarires needed
conda install ninja -y
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboardx yapf addict einops scipy termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric
pip install spconv-cu118
pip install laspy
# Install shapely to calculate IoU between polygons
pip install shapely

# Build and install PointOps library
cd ./libs/pointops
python setup.py install
cd ../..

# Build and install MultiScaleDeformableAttention library
cd ./pointcept/models/point_transformer_v3/ops
python setup.py install
cd ../../../..

# Install flash-attn 1.0.9 for turing arch. (It will take a few minutes)
MAX_JOBS=4 pip install flash-attn==1.0.9 --no-build-isolation

```
 
### Train
```
sh scripts/train.sh -g 1 -d nmg -c ptv3_nmg_base -n ptv3_nmg_base
```

### Test
```
sh scripts/test.sh -p python -g 1 -d nmg -n ptv3_nmg_base -w model_last
```
- 2D outputs are stored as pickle file under OBB path in log directory

### Visualization and Analysis
- please refer to `Evaluate_Utility-DINO_obb.ipynb`
