# NMG_Fusion
Point Cloud Multi Tasks(3D Semantics Segmentation and 2D Oriented Object detection)

# Codes
- I changed DINO into OBB Task to integrate it with Point Transformer V3.
 
# Installation (on cuda 11.x)

I used the below environment for Maryam’s fusion thing (working on both PTv3 and DINO based OBB)

### Create a conda virtual environment with Python 3.8 (Same with PTv3)
```
conda create -n dino_obb python=3.8 -y
conda activate dino_obb
# Install libarires needed (Same with PTv3)
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
# Build and install MultiScaleDeformableAttention library
cd models/dino/ops
python setup.py install
cd ../../..
```
 
### Train
```
sh scripts/DINO_train_nmg_R50.sh
sh scripts/DINO_train_nmg_swin.sh
```

### Test
```
sh scripts/DINO_eval_nmg_R50.sh
sh scripts/DINO_eval_nmg_swin.sh
```
- outputs are stored as pickle file in log directory

### Visualization and Analysis
- please refer to `Evaluate_Utility-DINO_obb.ipynb`
