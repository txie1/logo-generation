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


#### Unconditioned Model

<img width="552" alt="image" src="https://github.com/txie1/pic16b/assets/117710195/cde431b4-6cc9-415e-a346-1563774ad8fd">




#### Context-Conditioned Model

<img width="669" alt="image" src="https://github.com/txie1/pic16b/assets/117710195/1ca1c247-4470-46ee-9263-f560f029eb39">

To incorporate text prompts as an additional input to control the inference process, we adopt the simplified strategy of converting text into categories of multi-hot-encodings. Specifically, we first analyze the top-_n_ highest frequency keywords training dataset (e.g. ["modern", "minimalism", "black", "white", "inscription"]). Then map each text into **context**, which is an _n_-dimensional binary encoding vector (e.g. a text only containing "modern" is mapped to [1, 0, 0, 0, 0]).

Then the context encoder is turned into embeddings along with timesteps, starting at the down-sample layers. This is then followed by the typical UNet architecture with residual connections and Attention layers. 

One point to note is that during the training iterations, we also included the random masking-out of the contexts (ie. turns context vector into [0, 0, 0, 0, 0]). This is beneficial for the model to learn the true logo signals, without the influence of text description and features. It is also noticed that the quality of generated samples would improve after this implementation.

```
for epoch in range(1, num_epochs+1):
    ...
    for step, batch in enumerate(train_dataloader):
        ...

        # randomly mask out context
        context_mask = torch.bernoulli(torch.zeros(context.shape[0]) + 0.9).to(device)
        context = context * context_mask.unsqueeze(-1)
        ...
```

To generate images with a trained model, simply input the desired text prompts. The model will sample from the learned joint distribution of logos and context encoders, and iterate through 50 DDIM inference steps to output a denoise, coherent new logo!

### Usage

#### User Interface and Text Embedding Pipeline

To create a seamless experience for our target users, who might not have ample background knowledge of diffusion models, we constructed a user interface using the `ipywidgets` library. The interface prompts the user to input words specifying the style of the logo they are imagining, and outputs the words to functionalities further down the pipeline which utilizes the trained model to generate images.

To use the UI, first go to the `train_model.ipynb` file, navigate to the directory where you have saved the required `.py` files, and load your pre-trained model checkpoint as demonstrated below:

```
# Load pre-trained model
checkpoint = torch.load('ckpt_1500_25.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
```

Then, run the cell below and input the style you are looking for. Input one word at a time and hit `Confirm`.

```
# Utilize model
model_input = modelui(glove_vectors)
```

After entering all the prompt, hit the `Finish` button and run the following code

```
# Get processed words and generate image
processed_words = get_processed_words()
generate_image(processed_words,device,noise_scheduler,model,n_inference_timesteps)
```

A desired image should be generated with the current trained model checkpoint.


<a name="limit"></a>
## Limitations and Future Work
- Since a text prompt may contain multiple keywords, the use of multi-hot-encodings creates difficulties for the model to learn an accurate representation for each feature. In this case, the differences between styles might not be easily distinguishable. For further improvement, it is beneficial to 1). utilize one-hot-encoding and more distinct keywords, 2). use an encoding vector of larger dimensions to cover more features (though also increases the model complexity and thus computation resources required).
- Instead of hot-encoders, consider leveraging language models to create embeddings for text prompts directly. This should significantly improve the model performance and enable the generation of logos based on flexible user inputs and semantic meanings.
- Explore different options and locations for text and timestep embeddings, and how such modifications lead to changes in model performance.
- 



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

