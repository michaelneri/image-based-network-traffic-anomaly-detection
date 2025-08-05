import os
import numpy as np
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
from torchsummary import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score
import matplotlib.image

class AE_byhand(nn.Module):

    def __init__(self, channels, dim_input, kernel_size, Nfc, latent_dim, max_pooling_kernel = 2): 
        super(AE_byhand, self).__init__()
        """
        :channels (Tuple[int] or List[int]) : number of filters for each block conv (at the moment it need to contain 3 ints)
        :param dim_input (int): width and height of the squared image
        :param kernel_size (int): kernel size of convolutional layers
        :param Nfc (int): number of intermediate neurons before the latent space
        :param latent_dim (int): number of features of the latent space
        :param max_pooling_kernel Optional[int]: size of the downsample filter 
        """

        # Assign the hyperparameters as attributes of the model
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel

        # Definition of three encoder convolutional blocks 
        # Here I put how the image is processed
        # It starts with the shape [b, 1 , dim_input, dim_input] where b is the number of processed images for each batch (also called batch_size)

        # [b, 1, dim_input, dim_input]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels[0], kernel_size=self.kernel_size,  padding="same"), # "same" means that padding is added in order to maintain the same shape after the convolution
            #nn.BatchNorm2d(num_features=self.channels[0]), # WARNING!!!!! THIS LAYER IS PRESENT ONLY IN THE DOTS CKPT
            nn.ELU(), #non-linear activaction funzion (Exponential Linear Unit)
            # [b, channels[0], dim_input, dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channel[0], dim_input, dim_input]
            nn.MaxPool2d(self.max_pooling_kernel), # downsample

            # [b, channels[0], dim_input//2, dim_input//2]

            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2, dim_input//2]
            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2, dim_input//2]
            nn.MaxPool2d(self.max_pooling_kernel), # downsample
            # [b, channels[1], dim_input//4, dim_input//4]

            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            # [b, channels[2], dim_input//4, dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            # [b, channels[2], dim_input//4, dim_input//4]

            nn.Conv2d(in_channels = self.channels[2], out_channels = 1, kernel_size=1), # called also "projection"
            # [b, 1, dim_input//4, dim_input//4]
            nn.Flatten(start_dim = 1), 
            # [b, dim_input//4 * dim_input//4 ]
            nn.Linear(in_features = (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel) * (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel))), out_features = self.Nfc),
            nn.ELU(),
            # [b, Nfc ]
            nn.Linear(in_features = self.Nfc, out_features = self.latent_dim),
            # [b, latend_dim]
        )

        # Now we apply the reverse function
        # We start with the encoded data 
        # [b, latend_dim]
        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.latent_dim, out_features = self.Nfc),
            nn.ELU(),
            # [b, Nfc]
            nn.Linear(in_features = Nfc, out_features = (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel)) * (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel))),
            nn.ELU(),
            # [b, dim_input//4 * dim_input//4]

            nn.Unflatten(1, (1, (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel)), (self.dim_input // (self.max_pooling_kernel * self.max_pooling_kernel)))),
            # [b, 1, dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels = 1, out_channels = self.channels[2], kernel_size=1), # inverse projection
            nn.BatchNorm2d(num_features=self.channels[2]),
            # [b, channels[2], dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]), 
            nn.ELU(),
            # [b, channels[2], dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            nn.Upsample(scale_factor = self.max_pooling_kernel),
            # [b, channels[2], dim_input//2 , dim_input//2]


            nn.ConvTranspose2d(in_channels=self.channels[2], out_channels=self.channels[1], kernel_size=self.kernel_size, padding = self.kernel_size//2),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2 , dim_input//2]
            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            nn.Upsample(scale_factor = self.max_pooling_kernel),
            # [b, channels[1], dim_input , dim_input]


            nn.ConvTranspose2d(in_channels=self.channels[1], out_channels=self.channels[0], kernel_size=self.kernel_size, padding = self.kernel_size//2),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channels[0], dim_input , dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channels[0], dim_input , dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels = 1, kernel_size=self.kernel_size, padding="same"),
            # [b, 1, dim_input , dim_input] which is the original shape
            nn.Sigmoid(), # applied in order to have a binary image
        )

    def forward(self, x):
        # encode the images
        encoded = self.encoder(x)
        # and then decode them
        decoded = self.decoder(encoded)
        # the output will be two, the encoded and the reconstructed image
        return encoded, decoded



################# VARIATIONAL ENCODER ##########################
class VAE(AE_byhand):

    def __init__(self, channels, dim_input, kernel_size, Nfc, latent_dim, max_pooling_kernel = 2): 
        super(AE_byhand, self).__init__()
        """
        :channels (Tuple[int] or List[int]) : number of filters for each block conv (at the moment it need to contain 3 ints)
        :param dim_input (int): width and height of the squared image
        :param kernel_size (int): kernel size of convolutional layers
        :param Nfc (int): number of intermediate neurons before the latent space
        :param latent_dim (int): number of features of the latent space
        :param max_pooling_kernel Optional[int]: size of the downsample filter 
        """

        # Assign the hyperparameters as attributes of the model
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel

        # Definition of three encoder convolutional blocks 
        # Here I put how the image is processed
        # It starts with the shape [b, 1 , dim_input, dim_input] where b is the number of processed images for each batch (also called batch_size)

        # [b, 1, dim_input, dim_input]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels[0], kernel_size=self.kernel_size,  padding="same"), # "same" means that padding is added in order to maintain the same shape after the convolution
            #nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(), #non-linear activaction funzion (Exponential Linear Unit)
            # [b, channels[0], dim_input, dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channel[0], dim_input, dim_input]
            nn.MaxPool2d(self.max_pooling_kernel), # downsample

            # [b, channels[0], dim_input//2, dim_input//2]

            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2, dim_input//2]
            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2, dim_input//2]
            nn.MaxPool2d(self.max_pooling_kernel), # downsample
            # [b, channels[1], dim_input//4, dim_input//4]

            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            # [b, channels[2], dim_input//4, dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            # [b, channels[2], dim_input//4, dim_input//4]

            nn.Conv2d(in_channels = self.channels[2], out_channels = 1, kernel_size=1), # called also "projection"
            # [b, 1, dim_input//4, dim_input//4]
            nn.Flatten(start_dim = 1), 
            # [b, dim_input//4 * dim_input//4 ]
            nn.Linear(in_features = (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel) * (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel))), out_features = self.Nfc),
            nn.ELU(),
            # [b, Nfc ]
        )

        self.fc_mu = nn.Linear(in_features = self.Nfc, out_features = self.latent_dim)
        self.fc_logvar = nn.Linear(in_features = self.Nfc, out_features = self.latent_dim)


        # Now we apply the reverse function
        # We start with the encoded data 
        # [b, latend_dim]
        self.decoder = nn.Sequential(
            nn.Linear(in_features = self.latent_dim, out_features = self.Nfc),
            nn.ELU(),
            # [b, Nfc]
            nn.Linear(in_features = Nfc, out_features = (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel)) * (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel))),
            nn.ELU(),
            # [b, dim_input//4 * dim_input//4]

            nn.Unflatten(1, (1, (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel)), (self.dim_input // (self.max_pooling_kernel ** self.max_pooling_kernel)))),
            # [b, 1, dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels = 1, out_channels = self.channels[2], kernel_size=1), # inverse projection
            nn.BatchNorm2d(num_features=self.channels[2]),
            # [b, channels[2], dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]), 
            nn.ELU(),
            # [b, channels[2], dim_input//4 , dim_input//4]
            nn.Conv2d(in_channels=self.channels[2], out_channels=self.channels[2], kernel_size=self.kernel_size,  padding="same"),
            nn.BatchNorm2d(num_features=self.channels[2]),
            nn.ELU(),
            nn.Upsample(scale_factor = self.max_pooling_kernel),
            # [b, channels[2], dim_input//2 , dim_input//2]


            nn.ConvTranspose2d(in_channels=self.channels[2], out_channels=self.channels[1], kernel_size=self.kernel_size, padding = self.kernel_size//2),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            # [b, channels[1], dim_input//2 , dim_input//2]
            nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[1], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[1]),
            nn.ELU(),
            nn.Upsample(scale_factor = self.max_pooling_kernel),
            # [b, channels[1], dim_input , dim_input]


            nn.ConvTranspose2d(in_channels=self.channels[1], out_channels=self.channels[0], kernel_size=self.kernel_size, padding = self.kernel_size//2),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channels[0], dim_input , dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[0], kernel_size=self.kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=self.channels[0]),
            nn.ELU(),
            # [b, channels[0], dim_input , dim_input]
            nn.Conv2d(in_channels=self.channels[0], out_channels = 1, kernel_size=self.kernel_size, padding="same"),
            # [b, 1, dim_input , dim_input] which is the original shape
            nn.Sigmoid(), # applied in order to have a binary image
        )      

    def reparameterize(self, mu, logvar): # for LASSO 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        before_sampling = self.encoder(x)
        mu = self.fc_mu(before_sampling)
        logvar = self.fc_logvar(before_sampling)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar
        

################# AUTOENCODER MODULE #####################
class AEModule(LightningModule):
    def __init__(self, channels, dim_input, kernel_size, Nfc, latent_dim, max_pooling_kernel, lr, threshold, wBCE):
        super().__init__()
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel
        self.model = AE_byhand(channels = self.channels, dim_input = self.dim_input, kernel_size = self.kernel_size, Nfc = self.Nfc, latent_dim = self.latent_dim, 
                                max_pooling_kernel = self.max_pooling_kernel)
        #self.loss = nn.BCELoss()
        self.accuracy = Accuracy(task = "binary", num_classes=2, average="macro")
        self.test_precision = Precision(task = "binary", num_classes=2, average="macro")
        self.recall = Recall(task = "binary", num_classes=2, average="macro")
        self.F1Score = F1Score(task = "binary", num_classes=2, average="macro")
        self.lr = lr
        self.threshold = threshold
        self.errors_list = []
        self.wBCE = wBCE
        self.count = 0
        self.clean_errors = []
        self.anomalous_errors = []
        self.z_values_anomalous = []
        self.z_values_clean = []

    def weighted_BCELoss(self, pred, gt, batch_mean=False):
        weigth_map = torch.ones([gt.shape[0], 1, gt.shape[2], gt.shape[3]], device='cuda')
        weigth_map[gt != 0] = self.wBCE
        restore_error = -torch.mean(weigth_map * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12)),
                                dim=(1, 2, 3))
        ## Return a batch_size errors list (Calculate mean only on the last three dimensions)
        if batch_mean == True:
            return restore_error
        ## Calculate total mean of all errors to get a scalar value
        restore_error = torch.mean(restore_error)
        return restore_error
    
    def forward(self, x):
        return self.model.forward(x) # final

    def training_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)
        encoded, decoded = self(batch)
        # loss = self.loss(decoded, batch)
        loss = self.weighted_BCELoss(decoded, batch)

        if self.global_step % 500 == 0:
            path = os.path.join("Images_Train\\_save_{}".format(self.global_step))
            file_info = [
                "Reconstruction_error: {}".format(loss.item()),
            ]

            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

            matplotlib.image.imsave(os.path.join(path, "image.png"), batch.squeeze(1).tolist()[0])
            matplotlib.image.imsave(os.path.join(path, "image_rec.png"), decoded.squeeze(1).tolist()[0])

            with open(os.path.join(path, "info.txt"), 'w') as file:
                for line in file_info:
                    file.write(line)
                    file.write('\n')

        self.log("Train/loss", loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        image = image.unsqueeze(1)
        encoded, decoded = self(image)
        loss = self.weighted_BCELoss(decoded, image)

        pred = errors = self.weighted_BCELoss(decoded, image, batch_mean=True)
        self.clean_errors.append(pred[label == 0])
        self.anomalous_errors.append(pred[label == 1])
        self.z_values_anomalous.append(encoded[label == 1])
        self.z_values_clean.append(encoded[label == 0])
        pred = pred > self.threshold
        pred = pred.long()
        '''
        if self.count % 20 == 0:
            path = os.path.join("Images_Test\\_save_{}_{}".format(self.count, label.tolist()[0]))
            file_info = [
                "Image_label: {}".format(label.tolist()[0]), 
                "Reconstruction_error: {}".format(errors.tolist()[0]),
            ]

            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

            matplotlib.image.imsave(os.path.join(path, "image.png"), image.squeeze(1).tolist()[0])
            matplotlib.image.imsave(os.path.join(path, "image_rec.png"), decoded.squeeze(1).tolist()[0])

            with open(os.path.join(path, "info.txt"), 'w') as file:
                for line in file_info:
                    file.write(line)
                    file.write('\n')

        self.count += 1
        '''
        self.accuracy(pred, label)
        self.test_precision(pred, label)
        self.recall(pred, label)
        self.F1Score(pred, label)

        self.log("Test/accuracy", self.accuracy, on_epoch=True, on_step = False)
        self.log("Test/precision", self.test_precision, on_epoch=True, on_step = False)
        self.log("Test/recall", self.recall, on_epoch=True, on_step = False)
        self.log("Test/F1Score", self.F1Score, on_epoch=True, on_step = False)

    def validation_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)
        encoded, decoded = self(batch)
        # loss = self.loss(decoded, batch)
        loss = self.weighted_BCELoss(decoded, batch)

        ## Save a batch_size list of errors to calculate the best threshold later.
        restore_errors = self.weighted_BCELoss(decoded, batch, batch_mean=True)
        self.errors_list.append(restore_errors)

        self.log("Val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr)
        # return opt
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience = 2, factor = 0.1),
               "monitor": "Val/loss",
               "frequency": 1
                           },
              }
    

################## VARIATIONAL AUTOENCODER #############
class VAEModule(LightningModule):
    def __init__(self, channels, dim_input, kernel_size, Nfc, latent_dim, max_pooling_kernel, lr, threshold, wBCE):
        super().__init__()
        self.channels = channels
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.Nfc = Nfc
        self.latent_dim = latent_dim
        self.max_pooling_kernel = max_pooling_kernel
        self.model = VAE(channels = self.channels, dim_input = self.dim_input, kernel_size = self.kernel_size, Nfc = self.Nfc, latent_dim = self.latent_dim, 
                                max_pooling_kernel = self.max_pooling_kernel)
        #self.loss = nn.BCELoss()
        self.accuracy = Accuracy(task = "binary", num_classes=2, average="macro")
        self.test_precision = Precision(task = "binary", num_classes=2, average="macro")
        self.recall = Recall(task = "binary", num_classes=2, average="macro")
        self.F1Score = F1Score(task = "binary", num_classes=2, average="macro")
        self.lr = lr
        self.threshold = threshold
        self.errors_list = []
        self.wBCE = wBCE
        self.count = 0
        self.clean_errors = []
        self.anomalous_errors = []
        self.z_values_anomalous = []
        self.z_values_clean = []

    def weighted_BCELoss(self, pred, gt, batch_mean=False):
        weigth_map = torch.ones([gt.shape[0], 1, gt.shape[2], gt.shape[3]], device='cuda')
        weigth_map[gt != 0] = self.wBCE
        restore_error = -torch.mean(weigth_map * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12)),
                                dim=(1, 2, 3))
        ## Return a batch_size errors list (Calculate mean only on the last three dimensions)
        if batch_mean == True:
            return restore_error
        ## Calculate total mean of all errors to get a scalar value
        restore_error = torch.mean(restore_error)
        return restore_error
    
    def forward(self, x):
        return self.model.forward(x) # final

    def training_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)
        decoded, mu, logvar = self(batch)
        # loss = self.loss(decoded, batch)
        loss = self.weighted_BCELoss(decoded, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        final_loss = 0.00005*kl_loss + loss

        if self.global_step % 500 == 0:
            path = os.path.join("Images_Train\\_save_{}_ao".format(self.global_step))
            file_info = [
                "Reconstruction_error: {}".format(loss.item()),
            ]

            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

            matplotlib.image.imsave(os.path.join(path, "image_ao.png"), batch.squeeze(1).tolist()[0])
            matplotlib.image.imsave(os.path.join(path, "image_rec_ao.png"), decoded.squeeze(1).tolist()[0])

            with open(os.path.join(path, "info_ao.txt"), 'w') as file:
                for line in file_info:
                    file.write(line)
                    file.write('\n')

        self.log("Train/loss", final_loss, prog_bar=True)
        self.log("Train/kl", kl_loss )
        self.log("Train/rec_err", loss)

        return final_loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        image = image.unsqueeze(1)

        decoded, mu, logvar = self(image)

        loss = self.weighted_BCELoss(decoded, image, batch_mean = True)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
        loss = 0.00005*kl_loss + loss

        errors = loss
        pred = errors
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        self.z_values_anomalous.append(z[label == 1])
        self.z_values_clean.append(z[label == 0])
        self.clean_errors.append(pred[label == 0])
        self.anomalous_errors.append(pred[label == 1])

        pred = pred > self.threshold
        pred = pred.long()
        
        '''
        if self.count % 20 == 0:
            path = os.path.join("Images_Test\\_save_{}_{}".format(self.count, label.tolist()[0]))
            file_info = [
                "Image_label: {}".format(label.tolist()[0]), 
                "Reconstruction_error: {}".format(errors.tolist()[0]),
            ]

            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))

            matplotlib.image.imsave(os.path.join(path, "image.png"), image.squeeze(1).tolist()[0])
            matplotlib.image.imsave(os.path.join(path, "image_rec.png"), decoded.squeeze(1).tolist()[0])

            with open(os.path.join(path, "info.txt"), 'w') as file:
                for line in file_info:
                    file.write(line)
                    file.write('\n')

        self.count += 1
        '''
        self.accuracy(pred, label)
        self.test_precision(pred, label)
        self.recall(pred, label)
        self.F1Score(pred, label)

        self.log("Test/accuracy", self.accuracy, on_epoch=True, on_step = False)
        self.log("Test/precision", self.test_precision, on_epoch=True, on_step = False)
        self.log("Test/recall", self.recall, on_epoch=True, on_step = False)
        self.log("Test/F1Score", self.F1Score, on_epoch=True, on_step = False)

       

    def validation_step(self, batch, batch_idx):
        batch = batch.unsqueeze(1)
        decoded, mu, logvar = self(batch)
        # loss = self.loss(decoded, batch)
        loss = self.weighted_BCELoss(decoded, batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = 0.00005* kl_loss + loss

        ## Save a batch_size list of errors to calculate the best threshold later.
        restore_errors = self.weighted_BCELoss(decoded, batch, batch_mean=True)
        restore_errors = restore_errors + -0.5 * 0.00005* torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
        self.errors_list.append(restore_errors)

        self.log("Val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr = self.lr)
        # return opt
        return {
           "optimizer": opt,
           "lr_scheduler": {
               "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True, patience = 2, factor = 0.1),
               "monitor": "Val/loss",
               "frequency": 1
                           },
              }
    
## TEST OF THE DEFINITION AND FORWARD METHODS
if __name__ == "__main__":
    dim_input = 256 # since the images are 256x256

    # design of the model
    channels = [16, 32, 64]
    kernel_size = 5
    Nfc = 128
    latent_dim = 20
    max_pooling_kernel = 2
    # definition of the model
    model = AEModule(channels = channels, dim_input = dim_input, kernel_size = kernel_size, Nfc = Nfc, latent_dim = latent_dim, max_pooling_kernel = max_pooling_kernel, wBCE = 1, lr=0.01, threshold=10)

    # example of a batched input of images
    batch_size = 16
    test_images = torch.rand(batch_size, 1, dim_input, dim_input) # Note here the 1 at the channels position means that the image is gray-scale

    # try to feed these images to the model
    encoded, reconstructed = model(test_images)

    print("Size of the encoded image: {}".format(encoded.shape))
    print("Size of the reconstructed image: {}".format(reconstructed.shape))

    
    # fancy summary of the model with parameters and disk usage complexities
    summary(model, (1, dim_input, dim_input), device="cpu")

    ##n VARIATIONAL AUTOENCODER
        # definition of the model
    print("*************")
    print("VAE TEST")
    model = VAEModule(channels = channels, dim_input = dim_input, kernel_size = kernel_size, Nfc = Nfc, latent_dim = latent_dim, max_pooling_kernel = max_pooling_kernel, wBCE = 1, lr=0.01, threshold=10)

    # example of a batched input of images
    batch_size = 16
    test_images = torch.rand(batch_size, 1, dim_input, dim_input) # Note here the 1 at the channels position means that the image is gray-scale

    # try to feed these images to the model
    decoded, mu, log_var = model(test_images)

    print("Size of the encoded image mu: {}".format(mu.shape))
    print("Size of the encoded image log_var: {}".format(log_var.shape))
    print("Size of the decoded image: {}".format(decoded.shape))
    

    # fancy summary of the model with parameters and disk usage complexities
    summary(model, (1, dim_input, dim_input), device="cpu")


