
'''
Trains a PyTorch image classification model using device-agnostic code.
''' 

import os 
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer
from utils import fit_text
import argparse

# Getting arguments
parser = argparse.ArgumentParser()


data_group= parser.add_argument_group("Data")
data_group.add_argument("--data_path", type=str, default=".data", metavar="DIR",
                    help="Path where data is stored. (type: %(type)s, default: %(default)s)")
data_group.add_argument("--batch_size", type=int, default=32, metavar="BATCH",
                    help="Size of each batch in the dataloder. (type: %(type)s, default: %(default)s)")

train_group = parser.add_argument_group("Training")
train_group.add_argument("--epochs", type=int, default=5, metavar="EPOCHS",
                    help="Number of epochs to train the model. (type: %(type)s, default: %(default)s)")
train_group.add_argument("--learning_rate", type=float, default=0.001, metavar="LR",
                    help="Optimizer's learning rate. (type: %(type)s, default: %(default)s)")

model_group = parser.add_argument_group("Model")
model_group.add_argument("--hidden_units", type=int, default=10, metavar="H_UNITS",
                    help="Number of hidden units in the neural network. (type: %(type)s, default: %(default)s)")

options = parser.parse_args()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup hyperparameters
NUM_EPOCHS = options.epochs
BATCH_SIZE = options.batch_size
HIDDEN_UNITS = options.hidden_units
LEARNING_RATE = options.learning_rate
DATA_PATH = options.data_path

# Setup directories 
train_dir = DATA_PATH+"/pizza_steak_sushi/train"
test_dir = DATA_PATH+"/pizza_steak_sushi/test"

# Device agnostic-code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataloders & class names 
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                  test_dir=test_dir,
                                                                  transform=data_transform,
                                                                  batch_size=BATCH_SIZE)
# Create the model
model = model_builder.TinyVGG(in_c=3,
                              out_shape=len(class_names),
                              hidden_units=HIDDEN_UNITS).to(device)

# Setup loss & optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=model.parameters(),
                        lr=LEARNING_RATE)

# Print out hyperparameters
print("--------------------- Hyperparameters ---------------------")
print(f"{'Data path:':<17}{fit_text(DATA_PATH,15)}| {'Batch size:':<17}{fit_text(BATCH_SIZE,15)}")
print(f"{'Learning rate:':<17}{fit_text(LEARNING_RATE,15)}| {'Number of epochs:':<17}{fit_text(NUM_EPOCHS,15)}")
print(f"{'Hidden units:':<17}{fit_text(HIDDEN_UNITS,15)}")
print("-----------------------------------------------------------")


# Train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_t = timer()
results = engine.train(model=model,
                       loss_fn=loss_fn,
                       optimizer=optim,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       epochs=NUM_EPOCHS,
                       dev=device)
end_t = timer()
print(f"[INFO]: Training time: {end_t-start_t:.3f}")

# Save the model
utils.save_model(model=model,
                 target_dir="models",
                 model_name="0.5_going_modular_script_mode_TinyVGG.pth")

