# Direct Preference-based Policy Optimization without Reward Modeling (NeurIPS 2023)

Official implementation of [Direct Preference-based Policy Optimization without Reward Modeling](https://arxiv.org/abs/2301.12842), NeurIPS 2023.

## Installation 

Note: Our code was tested on Linux OS with CUDA 12. If your system specification differs (e.g., CUDA 11), you may need to modify the `requirements.txt` file and the installation commands.

Follow the steps below:
```
conda create -n dppo python=3.8
conda activate dppo

conda install -c "nvidia/label/cuda-12.3.0" cuda-nvcc
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## How to run DPPO

### Train preference model

```
python -m JaxPref.new_preference_reward_main --env_name [ENV_NAME] --seed [SEED] --transformer.smooth_w [NU] --smooth_sigma [M] 
```

### Train agent

```
python train.py --env_name [ENV_NAME] --seed [SEED] --transformer.smooth_w [NU] --smooth_sigma [M] --dropout [DROPOUT] --lambd [LAMBDA]
```

# Citation

 ```
@inproceedings{
    an2023dppo,
    title={Direct Preference-based Policy Optimization without Reward Modeling},
    author={Gaon An and Junhyeok Lee and Xingdong Zuo and Norio Kosaka and Kyung-Min Kim and Hyun Oh Song},
    booktitle={Neural Information Processing Systems},
    year={2023}
}
```

# Credit

Our code is based on [PreferenceTransformer](https://github.com/csmile-1006/PreferenceTransformer), which was also based on [FlaxModels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning).
