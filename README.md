
<div align="center">

<h1>Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction</h1>

<div>
    <a href='https://wutong16.github.io/' target='_blank'>Tong Wu</a>&emsp;
    <a href='https://myownskyw7.github.io/' target='_blank'>Jiaqi Wang</a>&emsp;
    <a href='https://xingangpan.github.io/' target='_blank'>Xingang Pan</a>&emsp;
    <a href='https://sheldontsui.github.io/' target='_blank'>Xudong Xu</a>&emsp;
    <a href='https://people.mpi-inf.mpg.de/~theobalt/' target='_blank'>Christian Theobalt</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=GMzzRRUAAAAJ&hl=zh-CN' target='_blank'>Dahua Lin</a>&emsp;
</div>

<strong>Accepted to <a href='https://iclr.cc/' target='_blank'>ICLR 2023</a> (Spotlight)</strong>

<strong><a href='https://arxiv.org/abs/2208.12697' target='_blank'>Paper</a></strong>
</div>


https://user-images.githubusercontent.com/28827385/222728479-af81dc68-6a15-4ab1-8632-5cbe3fcc17ad.mp4

## Updates
- [2023-03] Code released.
- [2023-01] :partying_face: Voxurf is accepted to ICLR 2023 (Spotlight)!

## Installation
Please first install a suitable version of [Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) on your machine. We tested on CUDA 11.1 with Pytorch 1.10.0.
```
git clone git@github.com/wutong16/Voxurf.git
cd Voxurf
pip install -r requirements.txt
```

## Datasets
### Public datasets
- [DTU](https://drive.google.com/file/d/1rAsmdno4v6X-HNDcwWaiJJXcpM4-aC3M/view?usp=share_link)
- [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) 
- [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip) 
- [BlendedMVS](https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip)
- [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
- [DeepVoxels](https://drive.google.com/open?id=1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH)
- [MobileBrick](https://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip)

Extract the datasets to `./data/`.

### Custom data
For your own data (e.g., a video or multi-view images), go through the preprocessing steps below. 
<details>
  <summary> Preprocessing (click to expand) </summary>
  
  - Please install [COLMAP](https://colmap.github.io/) and [rembg](https://github.com/danielgatis/rembg) first.
  
  - Extract video frames (if needed), remove the background, and save the masks.
```
mkdir data/<your-data-dir>
cd tools/preprocess
bash run_process_video.sh ../../data/<your-data-dir> <your-video-dir>
```

  - Estimate camera poses using COLMAP, and normalize them following [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md).
```
bash run_convert_camera.sh ../../data/<your-data-dir>
```

 - Finally, use `configs/custom_e2e` and run with `--scene <your-data-dir>`.
</details>


## Running
### Training
- You could find all the config files for the included datasets under `./configs`.
- To train on a set of images with a white/black background (recommended), use the corresponding config file and select a scene:
```
bash single_runner.sh <config_folder> <workdir> <scene>

# DTU example
bash single_runner.sh configs/dtu_e2e exp 122
```

- To train without foreground mask on DTU:
```
# DTU example
bash single_runner_womask.sh configs/dtu_e2e_womask exp 122
```

- To train without foreground mask on MobileBrick. The full evaluation on MobileBrick compared with other methods can be found [here](https://code.active.vision/MobileBrick/#:~:text=4.74-,Voxurf,-RGB).

```
# MobileBrick example
bash single_runner_womask.sh configs/mobilebrick_e2e_womask/ exp <scene>
```


### NVS evaluation
```
python run.py --config <config_folder>/fine.py -p <workdir> --sdf_mode voxurf_fine --scene <scene> --render_only --render_test
```

### Extracting the mesh & evaluation
```
python run.py --config <config_folder>/fine.py -p <workdir> --sdf_mode voxurf_fine --scene <scene> --render_only --mesh_from_sdf
```
Add `--extract_color` to get a **colored mesh** as below. It is out of the scope of this work to estimate the material, albedo, and illumination. We simply use the normal direction as the view direction to get the vertex colors.

![colored_mesh (1)](https://user-images.githubusercontent.com/28827385/222783393-63216e57-489c-46fb-9c24-4c8b6eed83bf.png)

## Citation
If you find the code useful for your research, please cite our paper.
```
@inproceedings{wu2022voxurf,
    title={Voxurf: Voxel-based Efficient and Accurate Neural Surface Reconstruction},
    author={Tong Wu and Jiaqi Wang and Xingang Pan and Xudong Xu and Christian Theobalt and Ziwei Liu and Dahua Lin},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2023},
}
```

## Acknowledgement 
Our code is heavily based on [DirectVoxGO](https://github.com/sunset1995/DirectVoxGO) and [NeuS](https://github.com/Totoro97/NeuS). Some of the preprocessing code is borrowed from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) and [LLFF](https://github.com/Fyusion/LLFF).
Thanks to the authors for their awesome works and great implementations! Please check out their papers for more details.

