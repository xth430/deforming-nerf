**Update:**
- `11/10/2022` We added a [manual](MANUAL.md) for training, mesh/cage generation and deformation, etc.

# Deforming Radiance Fields with Cages (ECCV 2022)

<img src="teaser.gif" width="100%">

[Project page](https://xth430.github.io/deforming-nerf/) | [Paper](https://arxiv.org/abs/2207.12298) | [Video](https://youtu.be/apE1q-_iQmQ)

## Setup
Create a conda environment:
```sh
conda env create -f environment.yml
conda activate deforming-nerf
```
Then install all dependencies (this may take a long time):
```
pip install .
```
Note that our code implementation is based on [Plenoxels](https://github.com/sxyu/svox2), if you have any installation problems, we recommend referring to their original repo.

## Data preparation

### Datasets
Download the dataset from the following links and put them under `./data/` directory:

**NeRF-synthetic** dataset: <https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1> (`nerf_synthetic.zip`)

**Synthetic-NSVF** dataset: <https://github.com/facebookresearch/NSVF#dataset> (`Synthetic_NSVF.zip`)

**DTU** dataset ([IDR](https://github.com/lioryariv/idr) preprocessed): <https://github.com/lioryariv/idr> (`DTU.zip`)

### Checkpoints
Download optimized Plenoxels models and preprocessed cages from [Google Drive](https://drive.google.com/drive/folders/1WiWjSteZ6sfkoVgvSeilGgOd3BDwlCNZ?usp=sharing) and put them under `.opt/ckpt/` directory.

## Rendering
Take `Lego` in NeRF-synthetic dataset as an example. 

Inside `opt/`, run the following command.

Render a static deformed scene with moving camera:
```
python render_imgs_deform.py ckpt/nerf_lego/ckpt.npz ../data/nerf_synthetic/lego/ -c configs/syn.json
```

Render an animation based on the cage interpolation:
```
python render_imgs_deform.py ckpt/nerf_lego/ckpt.npz ../data/nerf_synthetic/lego/ -c configs/syn.json --interpolate --cam_id 64
```

Render the original scene for comparison:
```
python render_imgs_deform.py ckpt/nerf_lego/ckpt.npz ../data/nerf_synthetic/lego/ -c configs/syn.json --render_orig
```

Note that you need to specify the config file according to the data format (i.e., `syn.json`, `syn_nsvf.json` or `dtu.json`)

## Custom datasets and cages
Please see [MANUAL.md](MANUAL.md) for some tips about:
- Optimizing Plenoxel model from scratch
- Extracting mesh from optimized Plenoxel model
- Generating cage from mesh
- Cage deformation

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{xu2022deforming,
      title={Deforming Radiance Fields with Cages}, 
      author={Tianhan Xu and Tatsuya Harada},
      year={2022},
      booktitle={ECCV},
}
```