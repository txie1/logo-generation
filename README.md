# Generative ML: Diffusion Models and Logo Generation

Contributors: Tong Xie, Janys Li, Judy Zhu


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">Overview</a>
      <ul>
        <li><a href="#built-with">Requirements</a></li>
      </ul>
      <ul>
        <li><a href="#dataset">Dataset</a></li>
      </ul>
    </li>
  <ol>
</details>





<a name="about-the-project"></a>
<!-- GETTING STARTED -->
## Overview

This project aims to explore and understand AI-generated art through a minimalist implementation of **diffusion models**. To this end, two models were trained from scratch with the purpose of generating new logos: 1). **Unconditioned model**, and 2). **Context-conditioned model**. A simple User-Interface is also designed to make the generation more convenient and accessible.

<a name="built-with"></a>
### Requirements (TODO: add packages)
- [PyTorch (version)](https://pytorch.org/get-started/locally/) -
- [Diffusers (0.24.0)](https://pypi.org/project/diffusers/) - State-of-the-art pretrained diffusion models
- 

<a name="dataset"></a>
### Dataset
- [Hugging Face modern-logo-dataset](https://huggingface.co/datasets/logo-wizard/modern-logo-dataset) - 803 pairs (x, y) where x is the image, y is the text description






## Resources Required

To complete this project, the following resources will be required:

- Data: A dataset of images to train the generative machine learning models. The dataset can be obtained from public sources such as Hugging Face.
- Computing Power: Sufficient computational resources will be needed to train and run the deep neural networks. This may include access to high-performance GPUs or cloud-based computing platforms.
- Software and Libraries: PyTorch framework, and other libraries and tools for data preprocessing, model evaluation, and visualization may also be needed.
Literature survey: Relevant research papers and documentation on generative models.

## Tools and Skills Required

The technical components of the project include:
Understanding of generative model architecture
Literature review: GAN, VAE, flow-based, diffusion models
Useful concepts: UNet, attention
(resources included below)
Implementation of models
utilizing the PyTorch framework and common datasets in literatures to implement and train the generative models
Python Packages: Torch, Numpy, Pandas, tqdm, SciPy, Pillow, Seaborn
Survey existing metrics used to measure model performance
Understanding of loss function and techniques such as early-stopping
Common metrics to measure generated images' quality (e.g. FID score)
Implementation of the metrics and critical analysis / discussion



## Risks & Ethics

Insufficient computational power required for training generative models 
Time required for training the deep neural networks
Collection of patent-free images that are big enough to be used for training
Generated images may contain sensitive / privacy-related content since the dataset used for training may not be fully regulated

## Tentative Timeline

- Week 4: confirmed project proposal and overall outline
- Week 6: completed data acquisition pipeline and model architecture
- Week 8: designed and implemented the text-to-image pipelines
- Week 10: completed training of the text-conditioned model & User-Interface


### Reference
- DeepLearning.AI: [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
- filipbasara0: [simple-diffusion](https://github.com/filipbasara0/simple-diffusion)
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239.pdf) 
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/pdf/2010.02502.pdf)
- [Lil's Log](https://lilianweng.github.io/)
- [Stanford CS229](https://cs229.stanford.edu/main_notes.pdf)
- [Deep Learning (Goodfellow)](https://www.deeplearningbook.org/)


