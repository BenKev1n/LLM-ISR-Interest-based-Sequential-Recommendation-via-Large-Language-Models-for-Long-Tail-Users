# LLM-ISR-Interest-based-Sequential-Recommendation-via-Large-Language-Models-for-Long-Tail-Users
This is the implementation of the paper "LLM-ISR: Interest-based Sequential Recommendation via Large Language Models for Long-Tail Users".

## Configure the environment

To ease the configuration of the environment, I list versions of my hardware and software equipments:

- Hardware:
  - GPU: Quadro RTX 8000 48GB
  - Cuda: 12.4
  - Driver version: 550.144.03
  - CPU: Intel(R) Xeon(R) Gold 6234
- Software:
  - Python: 3.9.5
  - Pytorch: 2.6.0+cu124

You can conda install the `environment.yml` or pip install the `requirements.txt` to configure the environment.

## Run and test

1. You can reproduce all LLM-ESR experiments by running the bash as follows:

```
bash experiments/yelp.bash
bash experiments/fashion.bash
bash experiments/beauty.bash
```

2. The log and results will be saved in the folder `log/`. The checkpoint will be saved in the folder `saved/`.

## Thanks

The code refers to the repo [LLM-ESR](https://github.com/liuqidong07/LLM-ESR/tree/master).
