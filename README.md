<div align="center">
  <h1>Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation</h1>


  <p align="center">
    <a href="https://arxiv.org/abs/2511.08935.pdf">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://mrwangyou.github.io/SCOPE/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>

  <a align="center" href="https://github.com/mrwangyou/SCOPE" target="_blank"><img width="850" src="assets/teaser.png"></a>
  
</div>

---


## Installation

You can set up the conda environment either by following the step-by-step instructions below or by using the provided `environment.yml` file.

### Option 1: Step-by-step installation (Linux, CUDA 11.8, Python 3.9.21)

```bash
conda create -n scope python=3.9.21 -y && conda activate scope

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2                                                      
```

### Option 2: Using environment.yml

We provide an `environment.yml` file for easy setup. This file includes all the dependencies (both conda and pip) and their versions.

```bash
conda env create -f environment.yml
conda activate scope
```

> Note: The `environment.yml` file was generated on a Linux system with CUDA 11.8. If you are using a different system, you may need to adjust the CUDA version or the packages accordingly.


## Run Evaluation

### 1 - Preparations

#### Dataset
Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify
the path in `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that 
contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, you can set the `scene_data_path` in the config files as `/your_path/hm3d/`.

#### OpenAI API Setup
Please set up the endpoint and API key for the OpenAI API in `src/const.py`.

<!-- ### 2 - Run Evaluation on A-EQA

First run the following script to generate the predictions for the A-EQA dataset:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
To split tasks, you can add `--start_ratio` and `--end_ratio` to specify the range of tasks to evaluate. For example,
to evaluate the first half of the dataset, you can run:
```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```
After the scripts finish, the results from all splits will be automatically aggregated and saved.

To evaluate the predictions with the pipeline from OpenEQA, you can refer to [link](https://github.com/yyuncong/3D-Mem-AEQA-Eval) -->

### 2 - Run Evaluation on GOAT-Bench
You can directly run the following script:
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```
The results will be saved and printed after the script finishes. 

## Acknowledgement

The codebase is built upon [3D-Mem](https://github.com/UMass-Embodied-AGI/3D-Mem), [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs).
We thank the authors for their great works.

## Citing SCOPE

```tex
@inproceedings{wang2026expand,
  title={Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation},
  author={Wang, Ningnan and Chen, Weihuang and Chen, Liming and Ji, Haoxuan and Guo, Zhongyu and Zhang, Xuchong and Sun, Hongbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={22},
  pages={18620--18628},
  year={2026}
}
```
