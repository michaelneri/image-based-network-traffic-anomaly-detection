import numpy as np
import torch
from data import CPSDatasetModule
from model import AEModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb


def find_threshold(errors_tensors_list, pfa):
    # Unisco la lista di tensori in un singolo tensore
    tensor = torch.cat(errors_tensors_list)
    # Trasformo il tensore in lista
    errors_list = tensor.tolist()

    # Ordino gli errori in ordine decrescente
    sorted_errors = sorted(errors_list, reverse=True)
    samples_num = int(np.multiply(len(errors_list), np.divide(pfa, 100)))
    
    # Voglio accettare samples_num campioni con errore maggiore del valore di threshold (permetto al modello di sbagliare 1 campione su 1000)
    # Ritorno quindi l'errore in posizione samples_num (i samples_num errori precedenti sono infatti maggiori)
    return sorted_errors[samples_num]


if __name__ == "__main__":

    wandb_logger = WandbLogger(project="ISeeYoo", name="test 25 dos53 pfa 0.01%")

    ## Def parametri (path, layers)
    filepath_train = "Dataset Tesi\\Training Set\\"
    filepath_test = "Dataset Tesi\\Test Set\\"
    ckpt_path = "./ISeeYoo/train_25/checkpoints/epoch=4-step=90000.ckpt"

    train_files_list = np.load("./connected_dots_split/train_files_list.npz")["arr_0"]
    val_files_list = np.load("./connected_dots_split/val_files_list.npz")["arr_0"]

    config = {
        "batch_size" : 16,
        "channels" : [16, 32, 64],
        "dim_input" : 256,
        "kernel_size" : 5,
        "Nfc" : 128,
        "latent_dim" : 20,
        "max_pooling_kernel" : 2,
        "lr" : 0.0001,
        "threshold" : 0,
        "wBCE" : 15
    }

    wandb_logger.log_hyperparams(config)

    model = AEModule.load_from_checkpoint(ckpt_path, channels = config["channels"], dim_input = config["dim_input"], kernel_size = config["kernel_size"], Nfc = config["Nfc"], latent_dim = config["latent_dim"], 
                    max_pooling_kernel = config["max_pooling_kernel"], lr=config["lr"], threshold=config["threshold"], wBCE=config["wBCE"])
    datamodule = CPSDatasetModule(file_path_training = filepath_train, file_path_test = filepath_test, train_files_list=train_files_list, val_files_list=val_files_list, batch_size=config["batch_size"])

    trainer = Trainer(accelerator="gpu", max_epochs = 5, logger=wandb_logger)
    wandb_logger.watch(model, log_graph=False)

    # trainer.validate(model=model, ckpt_path=ckpt_path, datamodule=datamodule)
    # model.threshold = find_threshold(errors_tensors_list=model.errors_list, pfa=0.01)
    # print(f"Final threshold = {model.threshold}")

    wandb_logger.log_hyperparams({"threshold":model.threshold})

    trainer.test(model, datamodule)

    wandb.finish()
    