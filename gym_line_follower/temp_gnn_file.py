import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.data import Data, DataLoader
import pickle
import os
import pdb
import numpy as np
from tqdm import tqdm
import random


# Define the Temporal GCN model
class TemporalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.linear(x)
        x = torch.relu(x)
        x = torch.sum(x, dim=0)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    # Define the path to save the model weights
    save_path = 'saved_gnn_models1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

    pkl_file_path = '/Users/aakamishra/school/cs329m/embedded-repair-mp/data_list1.pkl'
    if os.path.exists(pkl_file_path):
        with open(pkl_file_path, 'rb') as file:
            data_list = pickle.load(file)

    # Organize data into batches
    torch_data_list = []
    for i in range(0, len(data_list)):
        edge_lists, node_values, labels = data_list[i]
        #pdb.set_trace()
        if node_values.shape[1] == 11:
            data = Data(x=torch.tensor(node_values, dtype=torch.float32), 
                        edge_index=torch.tensor(np.vstack(edge_lists)))
            data.y = torch.tensor(labels, dtype=torch.float32)
            torch_data_list.append(data)

    # Define batch size
    batch_size = 32

    # Create DataLoader for batching the data
    loader = DataLoader(torch_data_list, batch_size=batch_size, shuffle=True, follow_batch=False)

    # Initialize the model, loss function, and optimizer
    model = TemporalGCN(input_dim=11, hidden_dim=256, output_dim=6)

    # # Path to your saved checkpoint
    # checkpoint_path = 'saved_gnn_models/model_epoch_20.pt'  # Replace with your checkpoint path

    # # Check if the checkpoint file exists
    # if os.path.exists(checkpoint_path):
    #     # Load the model checkpoint
    #     checkpoint = torch.load(checkpoint_path)

    #     # Load the model weights
    #     model.load_state_dict(checkpoint)
    #     print(f"Model loaded from checkpoint: {checkpoint_path}")
    # else:
    #     print(f"Checkpoint file '{checkpoint_path}' does not exist.")

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    random.seed(109)
    random.shuffle(torch_data_list)
    # Training loop
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        loss = 0
        for i in tqdm(range(len(torch_data_list))):
            data_point = torch_data_list[i]
            out = model(data_point.x, data_point.edge_index, data_point.batch)
            preds = torch.round(out)
            label = torch.ones(6) - data_point.y
            #pdb.set_trace()
            loss += criterion(preds, label)
            if (i + 1) % 32 == 0:
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