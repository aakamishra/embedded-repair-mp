import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGPooling, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import pickle
import os
import pdb
import numpy as np
from tqdm import tqdm
import random


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNN, self).__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = self.linear_final(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    # Define the path to save the model weights
    save_path = 'saved_dnn_models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    pkl_file_path = '/Users/aakamishra/school/cs329m/embedded-repair-mp/data_list_6_wheels.pkl'
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as file:
            data_list = pickle.load(file)
    # Organize data into batches
    torch_data_list = []
    for i in range(0, len(data_list)):
        edge_lists, node_values, labels = data_list[i]
        torch_data_list.append((node_values, labels))

    # Define batch size
    batch_size = 32

    # Create DataLoader for batching the data
    loader = DataLoader(torch_data_list, batch_size=batch_size, shuffle=True, follow_batch=False)

    # Initialize the model, loss function, and optimizer
    model = DNN(input_dim=50*139, hidden_dim=256, output_dim=6)

    # #Path to your saved checkpoint
    # checkpoint_path = 'saved_gnn_models1/model_epoch_70.pt'  # Replace with your checkpoint path

    # # Check if the checkpoint file exists
    # if os.path.exists(checkpoint_path):
    #     # Load the model checkpoint
    #     checkpoint = torch.load(checkpoint_path)

    #     # Load the model weights
    #     model.load_state_dict(checkpoint)
    #     print(f"Model loaded from checkpoint: {checkpoint_path}")
    # else:
    #     print(f"Checkpoint file '{checkpoint_path}' does not exist.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    random.seed(109)
    # Training loop
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loss = 0
        random.shuffle(torch_data_list)
        for i in tqdm(range(len(torch_data_list))):
            data_point, label = torch.tensor(torch_data_list[i][0]).float(), torch.tensor(torch_data_list[i][1]).float()
            out = model(data_point.flatten())
            # if (i + 1) % 128 == 0:
            #     pdb.set_trace()
            loss += criterion(out, label)
            if (i + 1) % 64 == 0:
                loss.backward()
                total_loss += loss.item()
                loss = 0
                optimizer.step()
                optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader)}")
        random.shuffle(torch_data_list)
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_path, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model weights saved at epoch {epoch}")
    # After training, you can use the trained model for inference
    # ... Perform inference using the trained model on new data