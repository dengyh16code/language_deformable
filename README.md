# Learning Language-Conditioned Deformable Object Manipulation with Graph Dynamics
**Yuhong Deng, Kai Mo, Chongkun Xia, Xueqian Wang**

**Tsinghua University**

This repository is a PyTorch implementation of the paper "Learning Language-Conditioned Deformable Object Manipulation with Graph Dynamics".

[Website](https://sites.google.com/view/language-deformable) | [ArXiv](https://arxiv.org/abs/2303.01310)

If you find this code useful in your research, please consider citing:

~~~
@misc{language_def,
      title={Learning Language-Conditioned Deformable Object Manipulation with Graph Dynamics}, 
      author={Kai Mo and Yuhong Deng and Chongkun Xia and Xueqian Wang},
      year={2023},
      eprint={2303.01310},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
~~~

## Table of Contents
* [Installation](#Installation)
* [Generate Data](#Generate-Data)
* [Train](#Train)
* [Evaluate](#Evaluate)

## Installation
This simulation environment is based on SoftGym. You can follow the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) to setup the simulator.

1. Clone this repository.

2. create a conda environment.
  ~~~
  conda env create -f environment.yml
  ~~~

3. Before you use the code, you should make sure the conda environment activated(`conda activate language_def`) and set up the paths appropriately: 
   ~~~
    export PYFLEXROOT=${PWD}/PyFlex
    export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
    export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
    export CLOTH3D_PATH=${PWD}/cloth3d
   ~~~
   The provided script `prepare_1.0.sh` includes these commands above.

## Generate Data

* Generate initial configurations:

  ~~~
  python generate_configs.py --cloth_type Square
  python generate_configs.py --cloth_type Rectangular
  python generate_configs.py --cloth_type Tshirt
  python generate_configs.py --cloth_type Trousers
  ~~~

  where `--cloth_type` specifies  the cloth type (Square | Rectangular | Tshirt | Trousers). These generated initial configurations will be saved in `configs/`

* Generate expert demonstrations:

  ```
  python generate_demo_fold.py --task CornerFold --cloth_type Square --randomize_pose
  python generate_demo_fold.py --task TriangleFold --cloth_type Square --randomize_pose
  python generate_demo_fold.py --task StraightFold --cloth_type Rectangular --randomize_pose
  python generate_demo_fold.py --task TshirtFold --cloth_type Tshirt --randomize_pose
  python generate_demo_fold.py --task TrousersFold --cloth_type Trousers --randomize_pose
  ```

  where `--task` specifies the task type (CornerFold | TriangleFold | StraightFold | TshirtFold | TrousersFold), `--random` specifies setting random pose as the intial state. These generated demonstrations will be saved in `raw_data`.


  `Demonstrator/demonstrator.py` includes the scripted demonstrator by accessing the ground truth position of each particle.

## Train 

* establish the training set:

  ```
  python create_dataset.py --task All --n_demos 1000
  ```

* Set up the model, optimizer and other details in yaml files in`train/train configs/`.

* Train model:

  ```
  python train.py --config_path depth_model
  python train.py --config_path depth_pre

  ```
  where `--config_path` specifies the `yaml` configuration filename in `train/train configs/`. The training process can be divided into two steps: train the
  action generator and success Classifier. 

## Evaluate 

* Evaluate the  model by running:

  ```
  python eval.py --task TrousersFold --cloth_type Trousers --model depth_model --agent_model epoch99 --predictor_model epoch99 --randomize

  ```

  The evaluation results are saved in `eval/`.

If you have any questions, please feel free to contact me via dengyh_work@outlook.com

