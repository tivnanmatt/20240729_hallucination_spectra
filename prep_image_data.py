import numpy as np
import torch 

# load the file TCGA-LIHC-numpy\TCGA-LIHC-numpy\training_volumes_32\TCGA_LIHC_000000.npy

# load the file TCGA-LIHC-numpy\TCGA-LIHC-numpy\testing_volumes_32\TCGA_LIHC_000000.npy

# load the file TCGA-LIHC-numpy\TCGA-LIHC-numpy\TCGA_LIHC_000000.npy

filename = '../../data/TCGA_LIHC/training_volumes_512/TCGA_LIHC_000000.npy'


for iFile in range(10,36):

    # filename = '../../data/TCGA_LIHC/training_volumes_512/TCGA_LIHC_00000' + str(iFile) + '.npy'
    # instead of this, use zfill to the appropriate number of zeros
    filename = '../../data/TCGA_LIHC/training_volumes_512/TCGA_LIHC_' + str(iFile).zfill(6) + '.npy'

    print('Loading file: ' + filename)

    data = np.load(filename)
    print('Processing data...')

    # cast data to np.float32
    data = data.astype(np.float32)

    # convert data to torch tensor
    new_data = torch.from_numpy(data)

    print('Saving data...')
    torch.save(new_data, 'TCGA_LIHC/training/training_TCGA_LIHC_' + str(iFile).zfill(6) + '.pt')

    del data
    del new_data

# # now do the testing_volumes into the testing folder and call the file names evaluation_
# filename = '../../data/TCGA_LIHC/testing_volumes_512/TCGA_LIHC_000000.npy'

# for iFile in range(9):
    
#         filename = '../../data/TCGA-LIHC-numpy/testing_volumes_512/TCGA_LIHC_' + str(iFile).zfill(6) + '.npy'
    
#         print('Loading file: ' + filename)
    
#         data = np.load(filename)
#         print('Processing data...')
    
#         # cast data to np.float32
#         data = data.astype(np.float32)
    
#         # convert data to torch tensor
#         new_data = torch.from_numpy(data)
    
#         print('Saving data...')
#         torch.save(new_data, 'TCGA_LIHC/testing/testing_TCGA_LIHC_' + str(iFile).zfill(6) + '.pt')
    
#         del data
#         del new_data




