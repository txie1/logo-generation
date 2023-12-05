<a name="readme-top"></a>

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
    <li><a href="#limit">Scope and Limitations</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Reference and Acknowledgments</a></li>
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
- [Datasets (2.15.0)](https://pypi.org/project/datasets/) - Data loader for public datasets
- [Numpy (1.21.5)](http://www.numpy.org/) - Multidimensional Mathematical Computing
- [Pandas (1.4.2)](https://pandas.pydata.org/docs/getting_started/overview.html#) - Access dataframes

<a name="dataset"></a>
### Dataset
- [Hugging Face modern-logo-dataset](https://huggingface.co/datasets/logo-wizard/modern-logo-dataset) - 803 pairs (x, y) where x is the image, y is the text description



## Example Usage
**(TODO: demo here)**


<img width="500" alt="image" src="https://github.com/txie1/pic16b/assets/117710195/acffc246-3e40-4f02-8a59-456f0bea58cb">

*Sample generated logos: resolution 32x32 (left), 64x64 (right)*



### Training

<img width="552" alt="image" src="https://github.com/txie1/pic16b/assets/117710195/cde431b4-6cc9-415e-a346-1563774ad8fd">

*(Unconditioned Model)*





<img width="669" alt="image" src="https://github.com/txie1/pic16b/assets/117710195/1ca1c247-4470-46ee-9263-f560f029eb39">

*(Context-conditioned Model)*





<a name="limit"></a>
## Limitations and Future Work




<a name="license"></a>
<!-- LICENSE -->
## License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



<a name="acknowledgments"></a>
<!-- ACKNOWLEDGMENTS -->
## Reference and Acknowledgment

- DeepLearning.AI: [How Diffusion Models Work](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
- filipbasara0: [simple-diffusion](https://github.com/filipbasara0/simple-diffusion)
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239.pdf) 
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/pdf/2010.02502.pdf)
- [Lil's Log](https://lilianweng.github.io/)
- [Stanford CS229](https://cs229.stanford.edu/main_notes.pdf)
- [Deep Learning (Goodfellow)](https://www.deeplearningbook.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

