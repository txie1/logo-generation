import torch
import numpy as np
import random
from datetime import datetime
import os
from PIL import Image

def set_seed(seed):
    """
    Set the seed for random number generators in Python, NumPy, and PyTorch to ensure reproducibility.

    This function sets the seed for random number generation.

    Parameters:
    seed (int): The seed value to be used for all random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        
def transforms(examples):
    """
    Process images and associated text for a machine learning model.

    This function applies image processing to each image in the input 'examples'
    and also performs multiple-hot-encoding on associated text based on specific keywords.

    Parameters:
    examples (dict): A dictionary containing 'image' and 'text' keys. 
                     'image' should be a list of image objects, and 'text' should be a list of strings.

    Returns:
    dict: A dictionary with keys 'input' and 'context'. 
          'input' contains processed images, and 'context' contains the multiple-hot-encoded text.
    """
    # Image processing: Convert each image to RGB and apply augmentations
    images = [
        augmentations(image.convert("RGB"))
        for image in examples["image"]
    ]

    # Define keywords for multiple-hot-encoding
    keywords = ["modern", "minimalism", "black", "white", "inscription"]

    # Multiple-hot-encoding of text: Create a tensor with 1s and 0s based on the presence of keywords
    contexts = torch.tensor([
        [1 if keyword in text.lower() else 0 for keyword in keywords]
        for text in examples["text"]
    ])

    # Return the processed images and contexts in a dictionary
    return {"input": images, "context": contexts}


def save_images(generated_images, epoch):
    """
    Save generated images to disk.

    This function processes the generated images and saves them individually, as well as in a grid format.
    The images are saved in a directory named after the current epoch and date-time.

    Parameters:
    generated_images (dict): A dictionary containing 'sample' and 'sample_pt' keys. 'sample' is expected to be
                             a list of image arrays, and 'sample_pt' is a PyTorch tensor of images.
    epoch (int): The current epoch number, used for naming the output directory.
    """
    # Extract images from the dictionary
    images = generated_images["sample"]

    # Convert images to a range of 0-255 and round them to the nearest integer
    images_processed = (images * 255).round().astype("uint8")

    # Get the current date and time for naming the output directory
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')

    # Create an output directory for the current epoch
    out_dir = f"./{samples_dir}/{epoch}/"
    os.makedirs(out_dir, exist_ok=True)

    # Save each image individually
    for idx, image in enumerate(images_processed):
        image = Image.fromarray(image)
        # Uncomment the line below to save each image individually
        # image.save(f"{out_dir}/{epoch}_{idx}.jpeg")

    # Save a grid of all generated images
    save_image(generated_images["sample_pt"], f"{out_dir}/{epoch}_grid.jpeg", nrow=eval_batch_size // 4)