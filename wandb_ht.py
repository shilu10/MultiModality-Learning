import wandb 

sweep_config = {
    'method': 'bayes',
    'name': 'hyperparameter-multimodal',
    'metric': {
        'goal': 'maximize', 
        'name': 'val_acc'
        },
    'parameters': {
        'epochs': {'values': [10, 15]},
        'lr': {'max': 00.1, 'min': 0.0001, 'distribution': 'uniform'},
        'optimizer': {
        'values': ['adam', 'sgd']
        },
     }
}


def sweep_train(config_defaults=None):
    # Initialize wandb with a sample project name
    with wandb.init(config=config_defaults):  # this gets over-written in the Sweep

        train(multimodal_model,  
              wandb.config.epochs,
              wandb.config.lr,
              BATCH_SIZE,
              wandb.config.optimizer,
              train_labels
             )