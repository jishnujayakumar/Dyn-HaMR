<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#get-started">Get Started</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
<br/>

## News :triangular_flag_on_post:
- [2025/06/04] Code released.
- [2024/12/18] [Paper](https://arxiv.org/abs/2412.12861) is now available on arXiv. ⭐

## Installation

### Environment setup
1. Clone the repository with submodules with the following command:
   ```bash
    git clone --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git
    cd Dyn-HaMR
    ```
    You can also run the following command to fetch the submodules:
    ```bash
    git submodule update --init --recursive .
    ```
  
2. To set up the virtual environment for Dyn-HaMR, we provide the integrated commands in `\scripts`. You can create the environment from
    ```bash
    source install_pip.sh
    ```

   Or, alternatively, create the environment from conda:   
    ```bash
    source install_conda.sh
    ```

### Model checkpoints download
Please run the following command to fetch the data dependencies. This will create a folder in `_DATA`:
  ```bash
  source prepare.sh
  ```
After processing, the folder layout should be:
```
|-- _DATA
|   |-- data/  
|   |   |-- mano/
|   |   |   |-- MANO_RIGHT.pkl
|   |   |-- mano_mean_params.npz
|   |-- BMC/
|   |-- hamer_ckpts/
|   |-- vitpose_ckpts/
|   |-- <SLAM model .pkl>
```

### Prerequisites
We use [MANO](https://mano.is.tue.mpg.de) model for hand mesh representation. Please visit the [MANO website](https://mano.is.tue.mpg.de) for registration and the model downloading. Please download `MANO_RIGHT.pkl` and put under the `_DATA/data/mano` folder.

## Get Started🚀

### Preparation
Please follow the instructions [here](https://github.com/MengHao666/Hand-BMC-pytorch) to calculate the below `.npz` files in order `dyn-hamr/optim/BMC/`:
```
|-- BMC
|   |-- bone_len_max.npy
|   |-- bone_len_min.npy
|   |-- CONVEX_HULLS.npy
|   |-- curvatures_max.npy
|   |-- curvatures_min.npy
|   |-- joint_angles.npy
|   |-- PHI_max.npy
|   |-- PHI_min.npy
```

> [!NOTE]
> If accurate camera parameters are available, please follow the format of `Dyn-HaMR/test/dynhamr/cameras/demo/shot-0/cameras.npz` to prepare the camera parameters for loading. Similarly, you can use Dyn-HaMR to refine and recover the hand mesh in the world coordinate system initializing from your own 2D & 3D motion data.

### Customize configurations
| Config | Operation |
|--------|-----------------|
| GPU | Edit in [`<CONFIG_GPU>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/config.yaml#L56) |
| Video info | Edit in [`<VIDEO_SEQ>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/data/video.yaml#L5) |
| Interval | Edit in [`<VIDEO_START_END>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/data/video.yaml#L16-L17) |
| Optimization configurations | Edit in [`<OPT_WEIGHTS>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/optim.yaml#L29-L49) |
| General configurations | Edit in [`<GENERAL_CONFIG>`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/dyn-hamr/confs/config.yaml) |

### Fitting on RGB-(D) videos 🎮
To run the optimization pipeline for fitting on arbitrary RGB-(D) videos, please first edit the path information here in `dyn-hamr/confs/data/video.yaml`, where `root` is the root folder to all of your datasets. `video_dir` is the corresponding folder that contains the videos. The key `seq` represents the video name you wanted to process. For example, you can run the following command to recover the global motion for `test/videos/demo.mp4`:

```
python run_opt.py data=video run_opt=True data.seq=demo
```
By default, the result will be saved to `outputs/logs/video-custom/<DATE>/<VIDEO_NAME>-<tracklet>-shot-<shot_id>-<start_frame_id>-<end_frame_id>`. After optimization, you can specify the output log dir and visualize the results by running the following command:
```
python run_vis.py --log_root <LOG_ROOT>
```
This will visualize all log subdirectories and save the rendered videos and images, as well as saved 3D meshes in the world space in `<LOG_ROOT>`. Please visit `run_vis.py` for further details. Alternatively, you can also use the following command to run and visualize the results in one-stage:
```
python -u run_opt.py data=video run_opt=True run_vis=True
```
As a multi-stage pipeline, you can customize the optimization process. Adding `run_prior=True` can activate the motion prior in stage III. Please note that in the current version, each motion chunk size needs to be set to 128 to be compatible with the original setting of HMP only when the prior module is activated.

### Blender Addon
Coming soon.

## Acknowledgements
The PyTorch implementation of MANO is based on [manopth](https://github.com/hassony2/manopth). Part of the fitting and optimization code of this repository is borrowed from [SLAHMR](https://github.com/vye16/slahmr). For data preprocessing and observation, [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [HaMeR](https://github.com/geopavlakos/hamer/) is used for 2D keypoints detection and MANO parameter initilization and [DPVO](https://github.com/princeton-vl/DPVO), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) is used for camera motion estimation. For biomechanical constraints and motion prior, we use the code from [here](https://github.com/MengHao666/Hand-BMC-pytorch) and [HMP](https://hmp.is.tue.mpg.de/). We thank all the authors for their impressive work!

## License
Please see [License](https://github.com/ZhengdiYu/Dyn-HaMR/blob/main/LICENSE) for details of Dyn-HaMR. This code and model are available only for non-commercial research purposes as defined in the LICENSE (i.e., MIT LICENSE). Note that, for MANO you must agree with the LICENSE of it. You can check the LICENSE of MANO from https://mano.is.tue.mpg.de/license.html.

## Citation
```bibtex
@article{yu2025dyn,
  title={Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera},
  author={Yu, Zhengdi and Zafeiriou, Stefanos and Birdal, Tolga},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
}
```

## Contact
For any technical questions, please contact z.yu23@imperial.ac.uk or ZhengdiYu@hotmail.com.
