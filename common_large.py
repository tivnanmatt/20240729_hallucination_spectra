import os
import torch
import laboratory as lab


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the components of the DiffusionBridgeModel
# same measurement_noise_variance for training and testing: changed from 0.1 to 0.02 to 0.00042375664 to 0.001695 (20 HU)
def get_diffusion_bridge_model(measurement_noise_variance=0.001695, train=True, num_files=16):
    # Dataset
    # change path



    # load data
    image_dataset = lab.torch.datasets.TCGA(
                            root="/mnt/CAMCA/home/alice",
                            train=train,
                            num_files=num_files,
                            verbose=True).to(device)
    
    # Measurement Noise Simulator
    measurement_simulator = lab.torch.distributions.gaussian.AdditiveWhiteGaussianNoise(
                                noise_variance=measurement_noise_variance).to(device)
    
    # Noise Variance Function
    def noise_variance_fn(t):
        if isinstance(t, int) or isinstance(t, float):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.view(-1, 1, 1, 1)
        return t * measurement_noise_variance

    # Noise Variance Derivative Function
    def noise_variance_prime_fn(t):
        t = t.view(-1, 1, 1, 1)
        return (0 * t + 1) * measurement_noise_variance

    # Forward SDE
    forward_SDE = lab.torch.sde.SongVarianceExplodingProcess(
                        noise_variance=noise_variance_fn, 
                        noise_variance_prime=noise_variance_prime_fn)

    # # Image Encoder
    # image_encoder = lab.torch.linalg.IdentityLinearOperator().to(device)

    # # Time Encoder
    # # change output_shape from (1, 28, 28) to (1, 64, 64)
    # time_encoder = lab.torch.networks.DenseNet(input_shape=(1,), 
    #                                            output_shape=(1, 64, 64), 
    #                                            hidden_channels_list=[782, 782], 
    #                                            activation='relu').to(device)

    # # Final Estimator
    # class FinalEstimator(torch.nn.Module):
    #     def __init__(self):
    #         super(FinalEstimator, self).__init__()
    #         self.densenet = lab.torch.networks.DenseNet(
    #                             input_shape=(2, 28, 28), 
    #                             output_shape=(1, 28, 28), 
    #                             hidden_channels_list=[782, 782, 782, 782], 
    #                             activation='relu')
    #         self.head = lab.torch.networks.DenseNet(
    #                         input_shape=(1, 28, 28), 
    #                         output_shape=(1, 28, 28), 
    #                         hidden_channels_list=[782], 
    #                         activation='linear')

    #     def forward(self, image_embedding, time_embedding):
    #         concatenated_data = torch.cat([image_embedding, time_embedding], dim=1)
    #         densenet_output = self.densenet(concatenated_data)
    #         return self.head(densenet_output)
    
    # final_estimator = FinalEstimator().to(device)




    # lets make a final estimator that does a convolutional u-net

    class Unet_FinalEstimator(torch.nn.Module):
        def __init__(self):
            super(Unet_FinalEstimator, self).__init__()
            # input shape is 2, 28, 28
            # output shape is 1, 28, 28

            class ConvBlock(torch.nn.Module):
                def __init__(self, in_channels, out_channels, activation='relu', batch_norm=True):
                    super(ConvBlock, self).__init__()
                    self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


                    if batch_norm:
                        self.bn = torch.nn.BatchNorm2d(out_channels)
                    else:
                        self.bn = torch.nn.Identity()

                    if activation == 'relu':
                        self.activation = torch.nn.ReLU()
                    elif activation == 'linear':
                        self.activation = torch.nn.Identity()
                def forward(self, x):
                    return self.activation(self.bn(self.conv(x)))

            class DownConvBlock(torch.nn.Module):
                def __init__(self, in_channels, out_channels, activation='relu', batch_norm=True):
                    super(DownConvBlock, self).__init__()
                    self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)



                    if batch_norm:
                        self.bn = torch.nn.BatchNorm2d(out_channels)
                    else:
                        self.bn = torch.nn.Identity()


                    if activation == 'relu':
                        self.activation = torch.nn.ReLU()
                    elif activation == 'linear':
                        self.activation = torch.nn.Identity()
                def forward(self, x):
                    return self.activation(self.bn(self.conv(x)))

            class UpConvBlock(torch.nn.Module):
                def __init__(self, in_channels, out_channels, activation='relu', batch_norm=True):
                    super(UpConvBlock, self).__init__()
                    self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, output_padding=1)
                    
                    if batch_norm:
                        self.bn = torch.nn.BatchNorm2d(out_channels)
                    else:
                        self.bn = torch.nn.Identity()

                    if activation == 'relu':
                        self.activation = torch.nn.ReLU()
                    elif activation == 'linear':
                        self.activation = torch.nn.Identity()

                    
                def forward(self, x):
                    return self.activation(self.bn(self.conv(x)))

            base_channels=16
            time_encoder_output_channels  = 15
            self.conv1 = ConvBlock(time_encoder_output_channels + 1, base_channels) # -> base_channels, 28, 28
            self.conv2 = ConvBlock(base_channels, base_channels) # -> base_channels, 28, 28
            self.down3 = DownConvBlock(base_channels, base_channels * 2) # -> base_channels * 2, 14, 14
            self.conv4 = ConvBlock(base_channels * 2, base_channels * 2) # -> base_channels * 2, 14, 14
            self.conv5 = ConvBlock(base_channels * 2, base_channels * 2) # -> base_channels * 2, 14, 14
            self.down6 = DownConvBlock(base_channels * 2, base_channels * 4) # -> base_channels * 4, 7, 7
            self.conv7 = ConvBlock(base_channels * 4, base_channels * 4) # -> base_channels * 4, 7, 7
            # self.flatten = torch.nn.Flatten()
            # self.fc1 = torch.nn.Linear(base_channels * 4 * 7 * 7, base_channels * 4 * 7 * 7)
            # self.fc2 = torch.nn.Linear(base_channels * 4 * 7 * 7, base_channels * 4 * 7 * 7)
            self.conv8 = ConvBlock(base_channels * 4, base_channels * 4) # -> base_channels * 4, 7, 7
            self.up9 = UpConvBlock(base_channels * 4, base_channels * 2) # -> base_channels * 2, 14, 14
            self.conv10 = ConvBlock(base_channels * 4, base_channels * 2) # -> base_channels * 2, 14, 14
            self.conv11 = ConvBlock(base_channels * 2, base_channels * 2) # -> base_channels * 2, 14, 14
            self.up12 = UpConvBlock(base_channels * 2, base_channels) # -> base_channels, 28, 28
            self.conv13 = ConvBlock(base_channels * 2, base_channels) # -> base_channels, 28, 28
            self.conv14 = ConvBlock(base_channels, base_channels) # -> base_channels, 28, 28
            self.conv15 = ConvBlock(base_channels, 1, activation='linear', batch_norm=False) # -> 1, 28, 28

            self.image_encoder =  lab.torch.linalg.IdentityLinearOperator().to(device)

            self.time_encoder = lab.torch.networks.DenseNet(input_shape=(1,), 
                                               output_shape=(time_encoder_output_channels,), 
                                               hidden_channels_list=[1024, 1024, 1024], 
                                               activation='relu').to(device)
            
            self.base_channels = base_channels
            self.time_encoder_output_channels = time_encoder_output_channels


        def forward(self, x_t, t):

            image_embedding = self.image_encoder(x_t)
            
            time_embedding = self.time_encoder(t)
            time_embedding = time_embedding.view(-1, self.time_encoder_output_channels, 1, 1)
            time_embedding = time_embedding.repeat(1, 1, image_embedding.shape[2], image_embedding.shape[3])

            concatenated_data = torch.cat([image_embedding, time_embedding], dim=1)
            x1 = self.conv1(concatenated_data)
            x2 = self.conv2(x1)
            x3 = self.down3(x2)
            x4 = self.conv4(x3)
            x5 = self.conv5(x4) 
            x6 = self.down6(x5)
            x7 = self.conv7(x6)
            x8 = self.conv8(x7)
            x9 = self.up9(x8)
            x10 = self.conv10(torch.cat([x9, x5], dim=1))
            x11 = self.conv11(x10)
            x12 = self.up12(x11)
            x13 = self.conv13(torch.cat([x12, x2], dim=1))
            x14 = self.conv14(x13)
            x15 = self.conv15(x14)

            # x_0_pred = x_t - (torch.sqrt(t.unsqueeze(-1).unsqueeze(-1))) * x15

            # x_0_pred = x_t - x15
            
            x_0_pred = x15

            return x_0_pred
        



    diffusion_backbone = Unet_FinalEstimator().to(device)



    diffusion_backbone = lab.torch.networks.DiffusersUnet2D(
                            input_channels=1,
                            time_encoder_hidden_size=256,
                            image_size=512,
                            unet_in_channels=16, 
                            unet_base_channels= 32,
                            unet_out_channels=1,
                            conditional_channels=0
                            )


    

    # Diffusion Model
    diffusion_model = lab.torch.diffusion.UnconditionalDiffusionModel(
                            forward_SDE=forward_SDE,
                            diffusion_backbone=diffusion_backbone,
                            estimator_type='noise').to(device)
    

    # Initial and Final Reconstructor
    initial_reconstructor = lab.torch.linalg.IdentityLinearOperator()
    final_reconstructor = lab.torch.linalg.IdentityLinearOperator()

    # Diffusion Bridge Image Reconstructor
    image_reconstructor = lab.torch.tasks.reconstruction.DiffusionBridgeImageReconstructor(
                            initial_reconstructor=initial_reconstructor,
                            diffusion_model=diffusion_model,
                            final_reconstructor=final_reconstructor
                            ).to(device)

    # Diffusion Bridge Model
    diffusion_bridge_model = lab.torch.tasks.reconstruction.DiffusionBridgeModel(
                                image_dataset=image_dataset,
                                measurement_simulator=measurement_simulator,
                                image_reconstructor=image_reconstructor,
                                task_evaluator='rmse'
                                ).to(device)

    return diffusion_bridge_model

def load_weights(diffusion_bridge_model, filename):
    if torch.cuda.is_available() and os.path.exists(filename):
        diffusion_bridge_model.image_reconstructor.diffusion_model.diffusion_backbone.load_state_dict(torch.load(filename))
    elif os.path.exists(filename):
        diffusion_bridge_model.image_reconstructor.diffusion_model.diffusion_backbone.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    else:
        print(f"Weights file '{filename}' not found.")

def save_weights(diffusion_bridge_model, filename):
    torch.save(diffusion_bridge_model.image_reconstructor.diffusion_model.diffusion_backbone.state_dict(), filename)
