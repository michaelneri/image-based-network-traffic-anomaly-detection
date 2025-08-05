import numpy as np
import torch
from data import CPSDatasetModule
from model import AEModule, VAEModule
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


def test_with_pfa(pfa, anomaly, is_validated, errors):
    if "scan11" in anomaly:
        attack = "scan11"
    elif "scan44" in anomaly:
        attack = "scan44"
    elif "dos11" in anomaly:
        attack = "dos11"
    elif "dos53" in anomaly:
        attack = "dos53"
    else:
        attack = "all attacks"
        

    run_name = "test conn_dots ideal {} pfa {}%".format(attack, pfa)
    wandb_logger = WandbLogger(project="ISeeYoo", name=run_name, tags = [attack, "conn_dots", "ideal"])

    ## Def parametri (path, layers)
    filepath_train = "Dataset Tesi\\Training Set\\"
    filepath_test = anomaly
    ckpt_path = "conn_dots_wbce_64.ckpt"

    train_files_list = np.load("./connected_dots_split/training.npz")["arr_0"]
    val_files_list = np.load("./connected_dots_split/validation.npz")["arr_0"]

    config = {
        "batch_size" : 64,
        "channels" : [16, 32, 64],
        "dim_input" : 256,
        "kernel_size" : 5,
        "Nfc" : 128,
        "latent_dim" : 64,
        "max_pooling_kernel" : 2,
        "lr" : 0.0001,
        "threshold" : 0,
        "wBCE" : 15,
    }
    wandb_logger.log_hyperparams(config)

    model = AEModule.load_from_checkpoint(ckpt_path, channels = config["channels"], dim_input = config["dim_input"], kernel_size = config["kernel_size"], Nfc = config["Nfc"], latent_dim = config["latent_dim"], 
                   max_pooling_kernel = config["max_pooling_kernel"], lr=config["lr"], threshold=config["threshold"], wBCE=config["wBCE"])
    datamodule = CPSDatasetModule(file_path_training = filepath_train, file_path_test = filepath_test, train_files_list=train_files_list, val_files_list=val_files_list, batch_size=config["batch_size"])

    trainer = Trainer(accelerator="gpu", gpus = 1, max_epochs = 5, logger=wandb_logger)
    wandb_logger.watch(model, log_graph=False)

    if not is_validated:
        trainer.validate(model=model, ckpt_path=ckpt_path, datamodule=datamodule)
        errors = model.errors_list
        is_validated = True
    model.threshold = find_threshold(errors_tensors_list=errors, pfa=pfa)
    print(f"Final threshold = {model.threshold}")

    wandb_logger.log_hyperparams({"threshold":model.threshold})

    trainer.test(model, datamodule)

    wandb.finish()
    return errors, is_validated


if __name__ == "__main__":

    # pfa_list = [5, 10, 14 , 18, 20]
    # pfa_list = [0.5, 1, 1.5, 2, 4, 6]
    pfa_list = [0.15]
    list_of_anomalies = ["Dataset Tesi\\Test All\\Test Set ConnDots\\Test all\\", "Dataset Tesi\\Test All\\Test Set ConnDots\\Test dos53\\", 
                        "Dataset Tesi\\Test All\\Test Set ConnDots\\Test dos11\\", 
                        "Dataset Tesi\\Test All\\Test Set ConnDots\\Test scan11\\", "Dataset Tesi\\Test All\\Test Set ConnDots\\Test scan44\\"]
    #list_of_anomalies = ["Dataset Tesi\\Test All\\Test Set Dots\\Test dos53\\", "Dataset Tesi\\Test All\\Test Set Dots\\Test dos11\\", 
    #                     "Dataset Tesi\\Test All\\Test Set Dots\\Test scan11\\", "Dataset Tesi\\Test All\\Test Set Dots\\Test scan44\\", 
    #                      "Dataset Tesi\\Test All\\Test Set Dots\\Test all\\"]
    errors_list = None
    is_validated = False
    for anomaly in list_of_anomalies:
        for pfa in pfa_list:
            errors_list, is_validated = test_with_pfa(pfa, anomaly, is_validated, errors_list)
    