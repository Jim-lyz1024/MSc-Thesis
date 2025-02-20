# Chapter 3: Domain Generalisation with Domain-Specific Knowledge
![ODE_framework](ODE_framework.png)

## Abstract
Many Domain Generalisation (DG) methods aim to learn domain-invariant representations from multiple source domains, enabling models to adapt to unseen target domains. While these methods have shown promise, they often neglect the value of domain-specific knowledge, which, though unique to each source domain, can still be relevant to unseen targets. We propose the Domain Description-Guided Experts (ODE) framework, which enhances DG by leveraging domain descriptions formulated as prompts incorporating domain names, to guide both the learning and utilization of domain-specific knowledge. During training, ODE employs parameter-efficient adapters as domain experts and leverages domain descriptions to guide domain-specific knowledge learning through text-image contrastive learning. These domain experts share a backbone network to maintain domain-invariant representations while learning unique domain characteristics. During inference, ODE leverages similarities between domain descriptions to determine the relevance of source domains to the target domain, guiding the integration of domain-specific knowledge. Experiments across seven benchmark datasets demonstrate the effectiveness of ODE, setting a new benchmark of 95.5% average accuracy on the PACS dataset.

## Baselines
DG approachs:
- ERM
- 2018 - CrossGrad - [Generalizing Across Domains via Cross-Gradient Training](https://openreview.net/forum?id=r1Dx7fbCW)
- 2020 - DDAIG - [Deep Domain-Adversarial Image Generation for Domain Generalisation](https://ojs.aaai.org/index.php/AAAI/article/view/7003)
- 2021 - MixStyle - [Domain Generalization with MixStyle](https://arxiv.org/abs/2104.02008#)
- 2022 - DomainMix - [Dynamic Domain Generalization](https://arxiv.org/abs/2205.13913)
- 2022 - EFDMix - [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Exact_Feature_Distribution_Matching_for_Arbitrary_Style_Transfer_and_Domain_CVPR_2022_paper.html)
- 2024 - SSPL - [Symmetric Self-Paced Learning for Domain Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/29639)

VLMs-based approachs:
- 2021 - CLIP-ZS - [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- 2024 - CLIP-A - [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://link.springer.com/article/10.1007/s11263-023-01891-x)

## Requirements
Python 3.8.19

Installing packages:
```pip install -r requirements.txt```

For GPU compatibility, please use CUDA 12.1.

## Run Commands
To run the code, use the following sample command:
```
python train.py
            --gpu 1                                                 # Specify device
            --seed 134                                              # Random Seed
            --output-dir output/ODE-RN50-NICO-autumn                # Output directory 
            --dataset NICO                                          # Specify dataset
            --source-domains dim grass outdoor rock water           # Source Domains
            --target-domains autumn                                 # Target Domain
            --model ODE                                             # Model for training
            --model-config-file config/ode.yaml                     # Config file for model
```