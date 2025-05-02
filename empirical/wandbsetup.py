import random

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="jassl-testing",
    # Set the wandb project where this run will be logged.
    project="jassl-team-surg",
   
    # Track hyperparameters and run metadata.
    # config={
    #     "learning_rate": 0.02,
    #     "architecture": "CNN",
    #     "dataset": "CIFAR-100",
    #     "epochs": 10,
    # },
)

# Simulate training.
epochs = 10