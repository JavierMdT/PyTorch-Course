from torchvision.io import read_image
from torchvision import transforms
from pathlib import Path
import argparse 
import torch
from model_builder import TinyVGG
from utils import fit_text

dev = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()

parser.add_argument("image_path", metavar="IMG", type=str,
                    help="Path to the image for which the prediction is desired.")
parser.add_argument("--model_path", metavar="MODEL", type=str ,
                    default="./models/0.5_going_modular_script_mode_TinyVGG.pth",
                    help="Path to the model that will make the inference.")
parser.add_argument("--hidden_units", metavar="UNITS", default=10, type=int,
                    help="Number of hidden units used in the model.")
parser.add_argument("image_class", metavar="CLASS", type=str,
                    help="Image class, it must be one between: [pizza, steak, sushi].")

options = parser.parse_args()
HIDDEN_UNITS = options.hidden_units
MODEL_PATH = options.model_path
IMAGE_PATH = options.image_path
IMAGE_CLASS = options.image_class

# Print out options
print("--------------------- Hyperparameters ---------------------")
print(f"{'Image path:':<17}{fit_text(IMAGE_PATH,15)}| {'Model path:':<17}{fit_text(MODEL_PATH,15)}")
print(f"{'Hidden units:':<17}{fit_text(HIDDEN_UNITS,15)}| {'Class:':<17}{fit_text(IMAGE_CLASS,15)}")
print("-----------------------------------------------------------")


################# Getting the image #################model_builder i
transform = transforms.Compose([
    transforms.Resize((64, 64)),
])
image = transform(read_image(IMAGE_PATH).type(dtype=torch.float32).to(dev)/255)

################# Getting the model #################
loaded__model = TinyVGG(in_c=3,
                        out_shape=3,
                        hidden_units=HIDDEN_UNITS)
state_dict = torch.load(MODEL_PATH)
loaded__model.load_state_dict(state_dict)

################# Making the prediction #################
class_names = ["pizza", "steak", "sushi"]
real_label = [IMAGE_CLASS]
print(f"Image shape: {image.shape}")
with torch.inference_mode():
    loaded__model.eval()
    logits = loaded__model(image.unsqueeze(dim=0))
    pred = torch.argmax(logits, dim=1).item()

################# Printing out #################
print(f"Real class: {IMAGE_CLASS} | Prediction: {class_names[pred]}")
