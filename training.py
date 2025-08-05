import numpy as np
from data import CPSDatasetModule
from model import AEModule, VAEModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb


if __name__ == "__main__":

    wandb_logger = WandbLogger(project="ISeeYoo", name="train dots VAE wBCE", tags = ["dots", "VAE", "wBCE"])

    ## Def parametri (path, layers)
    filepath_train = "Dataset Tesi\\Training Set\\"
    filepath_test = "Dataset Tesi\\Test Set\\"
    # ckpt_path = "./ISeeYoo/conn_dots_train_20_wbce/checkpoints/epoch=9-step=180000.ckpt"

    train_files_list = np.load("./dots_split/training.npz")["arr_0"]
    val_files_list = np.load("./dots_split/validation.npz")["arr_0"]

    config = {
        "batch_size" : 64,
        "channels" : [16, 32, 64],
        "dim_input" : 256,
        "kernel_size" : 5,
        "Nfc" : 256,
        "latent_dim" : 64,
        "max_pooling_kernel" : 2,
        "lr" : 0.0001,
        "threshold" : 0,
        "wBCE" : 15,
    }

    wandb_logger.log_hyperparams(config)

    # model = AEModule(channels = config["channels"], dim_input = config["dim_input"], kernel_size = config["kernel_size"], Nfc = config["Nfc"], latent_dim = config["latent_dim"], 
    #                 max_pooling_kernel = config["max_pooling_kernel"], lr=config["lr"], threshold=config["threshold"], wBCE=config["wBCE"])
    #model = VAEModule.load_from_checkpoint(checkpoint_path = "test.ckpt", channels = config["channels"], dim_input = config["dim_input"], kernel_size = config["kernel_size"], Nfc = config["Nfc"], latent_dim = config["latent_dim"], 
    #                max_pooling_kernel = config["max_pooling_kernel"], lr=config["lr"], threshold=config["threshold"], wBCE=config["wBCE"])
    model = VAEModule(channels = config["channels"], dim_input = config["dim_input"], kernel_size = config["kernel_size"], Nfc = config["Nfc"], latent_dim = config["latent_dim"], 
                    max_pooling_kernel = config["max_pooling_kernel"], lr=config["lr"], threshold=config["threshold"], wBCE=config["wBCE"])

    datamodule = CPSDatasetModule(file_path_training = filepath_train, file_path_test = filepath_test, train_files_list=train_files_list, val_files_list=val_files_list, batch_size=config["batch_size"])

    trainer = Trainer(accelerator="gpu", gpus = 1, max_epochs = 30, logger=wandb_logger)
    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule)

    wandb.finish()