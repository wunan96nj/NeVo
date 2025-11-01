from ac_dc.ncvv_ac_dc import ac_dc_encode, ac_dc_decode, ac_dc_encode2, ac_dc_decode2
import numpy as np
from codec import dct_3d,idct_3d,quantize_quality,gen_3d_quant_tbl,Timer,merge_volume,zero_unpads
from codec import split_volume,zero_pads,get_origin_size
import os
from bitarray import bitarray
import json
from codec import  recover_misc,recover_misc_deform, decode_jpeg_huffman,decode_entropy_motion_npy, unproject_pca_mmap
import torch

import os
import numpy as np
from bitarray import bitarray
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.nn as nn
import torch.optim as optim


print(torch.__version__)

QTY_3d =gen_3d_quant_tbl()

def decode_jpeg_huffman(file_object,header,device):

    data_size = header['size']
    quality = header['quality']

    rec=np.zeros(data_size, dtype=np.int16)
    ac_dc_decode2(rec, 0, data_size[0], data_size[1], file_object)

    rec = torch.tensor(rec, device=device).to(torch.float)
    quant_table = quantize_quality(QTY_3d, quality)
    rec=rec*quant_table

    rec = idct_3d(rec, norm='ortho')
    return rec

def recover_residual(residual_rec_dct, mask, n_channel):
    residual_rec=torch.zeros((mask.size(0),n_channel,voxel_size,voxel_size,voxel_size),device=device)
    residual_rec[mask] = residual_rec_dct
    #print(residual_rec.shape)
    residual_rec = residual_rec.reshape(mask.size(0), n_channel, -1)
    #print(residual_rec.shape)
    torch.cuda.empty_cache()
    return residual_rec


# Function to decode entropy motion numpy array (your provided function)
def decode_entropy_motion_npy(deform, deform_mask, device=torch.device('cuda')):
    rec = np.zeros((deform_mask.shape[0], 3))
    rec[deform_mask] = deform
    rec = torch.tensor(rec, device=device).to(torch.float)
    return rec

# Function to read a single frame's data
def read_motion_frame_data(frame_id):
    mask_size_8 = 30600  # Adjust as necessary
    mask_size = 30600  # Adjust as necessary

    if (frame_id % 20 == 0):
        frame_id = frame_id - 1
        
    if (not os.path.exists(os.path.join(file_path, f'deform_mask_{frame_id}.rerf'))):
        deform_rec = np.zeros((mask_size_8, 3))
        deform_rec = torch.tensor(deform_rec, device=torch.device('cuda')).to(torch.float)
        return deform_rec

    
    with open(os.path.join(file_path, f'deform_mask_{frame_id}.rerf'), 'rb') as masked_file:
        mask_bits = bitarray()
        mask_bits.fromfile(masked_file)
        deform_mask = np.unpackbits(mask_bits).reshape(mask_size_8)[:mask_size].astype(bool)

    deform = np.load(os.path.join(file_path, f'deform_{frame_id}.npy'))
    deform_rec = decode_entropy_motion_npy(deform, deform_mask, torch.device('cuda'))

    deform_rec = torch.tensor(deform_rec, dtype=torch.float32)
    
    return deform_rec.cpu()

# frame_count = 0
# frame_id = 0
group_size = 20
file_path = "./compressed_kpop"
pca = True
device = torch.device('cuda')
voxel_size = 8
n_channel = 12 + 1
def read_frame_data(frame_id):
    if (frame_id % group_size == 0):
        frame_id = frame_id - 1
    key_frame = (frame_id % group_size == 0)
    print(frame_id, key_frame)

    jsonfile=os.path.join(file_path,f'header_{frame_id}.json')
    with open(jsonfile) as f:
        headers = json.load(f)
        header = headers["headers"][0]
        mask_size = header['mask_size']
        if mask_size % 8 != 0:
            mask_size_8 = (mask_size // 8 + 1) * 8
        else:
            mask_size_8 = mask_size

    with open(os.path.join(file_path, f'mask_{frame_id}.rerf'), 'rb') as masked_file:
        mask_bits = bitarray()
        mask_bits.fromfile(masked_file)
        mask=torch.from_numpy(np.unpackbits(mask_bits).reshape(mask_size_8)[:mask_size].astype(bool)).cuda()
        #print(f"Number of elements in mask that are 1 or True: {mask.sum().item()}")


    quality=header["quality"]
    if not key_frame:
        rec_feature = []
        rec_feature.append(decode_jpeg_huffman(
            os.path.join(file_path, f'feature_{frame_id}_{quality}.rerf'), headers["headers"][0], device=device))
        rec_feature.append(decode_jpeg_huffman(
            os.path.join(file_path, f'feature_{frame_id}_{quality-1}.rerf'), headers["headers"][1], device=device))
        residual_rec_dct = unproject_pca_mmap(rec_feature, file_path, frame_id, device,voxel_size)
    else:
        residual_rec_dct = decode_jpeg_huffman(
            os.path.join(file_path, f'feature_{frame_id}_{quality}.rerf'), header, device=device)

    residual_rec = recover_residual(residual_rec_dct, mask, n_channel)
    del residual_rec_dct
    #print(residual_rec)
    torch.cuda.empty_cache()
    return residual_rec.cpu()

    
def custom_collate_fn(batch):
    # Filter out all the None items
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Signal to skip this batch
    # Otherwise, default collate. Use torch.utils.data.dataloader.default_collate to handle normal cases
    return torch.utils.data.dataloader.default_collate(batch)

class VoxelFrameDataset(torch.utils.data.Dataset):
    def __init__(self, points_per_frame=30600, history_length=9, total_frames=29):
        self.history_length = history_length
        # Assuming self.frames is populated as before, each with shape [30600, 3]
        self.total_points = points_per_frame * (total_frames - history_length - 1) * 512
        self.frame_id = -1
        self.frames = []

    def __len__(self):
        return self.total_points


    def __getitem__(self, idx):
        frame_idx = idx // (30600 * 512) + 11
        point_idx = idx % (30600 * 512) // 512
        feature_id = idx % 512
        if frame_idx != self.frame_id:
            self.frame_id = frame_idx
            


            self.frames = [read_frame_data(i) for i in range(frame_idx, frame_idx + self.history_length+1)]
            self.frames_motion = [read_motion_frame_data(i) for i in range(frame_idx, frame_idx + self.history_length+1)]

        # Convert the flattened point index to 3D coordinates
        z = point_idx // (30 * 30)
        y = (point_idx % (30 * 30)) // 30
        x = point_idx % 30

        #target_frame = self.frames[frame_idx + self.history_length][point_idx]
        target_frame = self.frames[self.history_length][point_idx][:, feature_id]
        # Assuming target_frame is a tensor of appropriate dimensions
        target_frame_motion = self.frames_motion[self.history_length][point_idx]
        target_frame = target_frame.flatten()
        if torch.all(torch.eq(target_frame, torch.zeros(13, device='cpu'))):
            return None  # Skip this item



        historical_frames = []
        for i in range(self.history_length):
            frame = self.frames[i]
            #neighbor_frames = []
            neighbor_frames = frame[point_idx][:, feature_id]

            historical_frames.append(neighbor_frames)

        historical_frames = torch.stack(historical_frames)
        #print(historical_frames.type, target_frame.type)
        torch.cuda.empty_cache()
        return historical_frames.cuda(), target_frame.cuda()

    
class VoxelPredictor(nn.Module):
    def __init__(self):
        super(VoxelPredictor, self).__init__()
        self.input_dim = 13  # Number of features at each time step
        self.hidden_dim = 512  # Hidden dimension of the LSTM
        self.num_layers = 2    # Number of LSTM layers

        # LSTM layer for processing sequences of voxel data
        self.lstm = nn.LSTM(input_size=self.input_dim, 
                            hidden_size=self.hidden_dim, 
                            num_layers=self.num_layers, 
                            batch_first=True)  # batch_first=True means the input tensors should be of shape (batch, seq, feature)

        # Output layer
        self.fc = nn.Linear(self.hidden_dim, 13)  # Output the same dimension as the input feature

    def forward(self, x):
        # Expecting x of shape: (batch_size, 9, 512)
        lstm_out, _ = self.lstm(x)  # Outputs from all timesteps
        x = lstm_out[:, -1, :]  # Take the output of the last timestep
        x = self.fc(x)  # Get final output predictions
        #x = torch.tanh(x)  # Ensure output is between -1 and 1
        return x


dataset = VoxelFrameDataset()


train_loader = DataLoader(dataset, batch_size=2048, collate_fn=custom_collate_fn, shuffle=False, drop_last=True)

model = VoxelPredictor().to(torch.device('cuda'))

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_l1 = nn.L1Loss()

# Training loop
def train_model(model, train_loader, epochs=30):
    model.train()
    for epoch in range(epochs):

        # Example training loop iteration
        #for inputs, targets in train_loader:
        for i, data in enumerate(train_loader):
            if data is None:
                continue  # Skip the batch if the collate_fn returned None
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)  # outputs shape will be [batch_size, 3]
            loss = criterion(outputs, targets)  # Ensure targets is [batch_size, 3]
            loss.backward()
            l1loss = criterion_l1(outputs, targets)
            optimizer.step()
            print(epoch, i, l1loss)
            torch.save(model, 'model.pth')
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train_model(model, train_loader, epochs=60)
