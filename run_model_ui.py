import gensim.downloader
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
from datetime import datetime
from PIL import Image
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


def get_closest_words(input_words,glove_vectors):
  '''
    This function takes in a list of words, and finds the closest word
    from the target list of words based on embedding distances.
  
    Parameters:
        input_words (list): A list of words.
        glove_vectors (dict): A dictionary mapping words to their embeddings.
    
    Returns:
        list: A list of closest words.
  '''
  

  # Embed target words
  target_words = ["modern", "minimalism", "black", "white", "inscription"]
  target_embedcoords = {word: glove_vectors[word] for word in target_words if word in glove_vectors}
  # target_embeddings

  # Embed input words
  words_embedcoords = {word: glove_vectors[word] for word in input_words if word in glove_vectors}

  # Calculate distances
  distances = {}
  for word, embedding in words_embedcoords.items():
      word_distances = {}
      for target_word, target_embedding in target_embedcoords.items():
          distance = np.linalg.norm(embedding - target_embedding)
          word_distances[target_word] = distance
      distances[word] = word_distances

  distances = pd.DataFrame(distances)
  closest = distances.idxmin()
  outputs = closest.values
  return outputs


# Global variable to store the processed words
processed_words_global = []

def modelui(glove_vectors):
    '''
    Generates a UI for the user to enter words and generate images.

    Parameters:
        None

    Returns:
        None
    '''
    global processed_words_global

    # Text label at the top
    top_text = widgets.HTML(value="<h3>Enter the words you want to create the logo with:</h3>")

    # Text input for the user to enter a word
    word_input = widgets.Text(placeholder='Enter a word here')

    # Output widget to display selected words
    output = widgets.Output()

    # List to keep track of selected words
    selected_words = []

    # Function to handle confirm button click
    def on_confirm_clicked(b):
        '''
        Handles the confirm button click.
        '''
        with output:
            word = word_input.value.strip()
            if word and word not in selected_words:
                selected_words.append(word)
                clear_output(wait=True)
                print("Selected Words:", ', '.join(selected_words))
            word_input.value = ''  # Clear the input field

    # Confirm button
    confirm_button = widgets.Button(description="Confirm")
    confirm_button.on_click(on_confirm_clicked)

    # Function to handle finish button click
    def on_finish_clicked(b):
        '''
        Handles the finish button click.
        '''
        global processed_words_global
        with output:
            clear_output(wait=True)
            print("Final Selected Words:", ', '.join(selected_words))
            processed_words_global = get_closest_words(selected_words, glove_vectors)
            print(f'These words are mapped to: {processed_words_global}')
            print('Sending to model!')

    # Finish button
    finish_button = widgets.Button(description="Finish")
    finish_button.on_click(on_finish_clicked)

    # Display the input field, buttons, and output
    display(top_text, word_input, confirm_button, finish_button, output)

def get_processed_words():
    '''
    Returns the processed words.
    
    Parameters:
        None
    Returns:
        list: A list of processed words.
    '''
    return processed_words_global

def generate_image(processed_words,device,noise_scheduler,model,n_inference_timesteps):
    '''
    Generate images based on the given checkpoint and processed words.

    Parameters:
        checkpoint (str): The path to the checkpoint file.
        processed_words (list): A list of processed words.

    Returns:
        numpy.ndarray: An array of generated images.
    '''
    def map_keywords_to_context_vector(keywords, keyword_to_position):
        '''
        Map keywords to a context vector based on a given keyword-to-position mapping.

        Parameters:
            keywords (list): A list of keywords.
            keyword_to_position (dict): A dictionary mapping keywords to their positions.

        Returns:
            list: A context vector representing the presence of keywords.
        '''
        context_vector = [0] * len(keyword_to_position)  # Initialize vector with zeros
        for keyword in keywords:
            if keyword in keyword_to_position:
                position = keyword_to_position[keyword]
                context_vector[position] = 1  # Set to 1 if keyword is present
        return context_vector

    # Define your keywords and their mapping
    keyword_to_position = {
        "modern": 0,
        "minimalism": 1,
        "black": 2,
        "white": 3,
        "inscription": 4
    }

    # Map the processed words to the context vector
    context_vector = map_keywords_to_context_vector(processed_words, keyword_to_position)

    # Set a seed for reproducibility
    seed = 7
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CUDA-enabled GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Convert context vector to tensor
    ctx = torch.tensor([context_vector], dtype=torch.float32).to(device)

    # Generate images with trained model
    with torch.no_grad():
        generated_images = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=None,
            eta=0.5,
            use_clipped_model_output=True,
            batch_size=len(ctx),
            output_type="numpy",
            device=device,
            context=ctx)

        images = generated_images["sample"]
        all_samples = torch.from_numpy(images)
        images_processed = (images * 255).round().astype("uint8")

        # Display each generated image
        for idx, image in enumerate(images_processed):
            plt.figure(figsize=(1, 1))
            plt.imshow(image)
            plt.axis('off')  # Hide the axis
            plt.show()
    return images