import numpy as np
import torch.nn as nn
import torch
from Mmodel import Mmodel
from MCPDataset import MCPDataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, r2_score, \
    mean_squared_error, mean_absolute_error

# %%
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 300
LR = 0.001
LOG_INTERVAL = 20
modeling = Mmodel
cuda_name = "cuda:0"

def train(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data["graphImg"] = data["graphImg"].float().to(device).permute(0, 3, 1, 2)
        data["ecfp"] = data["ecfp"].float().to(device)
        data["hash"] = data["hash"].float().to(device)
        data["list_of_node"] = data["list_of_node"].float().to(device)
        data["list_of_edge"] = data["list_of_edge"].float().to(device)
        data["list_of_pos"] = data["list_of_pos"].float().to(device)
        data["label"] = data["label"].float().to(device)
        optimizer.zero_grad()
        output = model(data["graphImg"],data["ecfp"],data["hash"],
                       data["list_of_node"],data["list_of_edge"],data["list_of_pos"])
        preds = output
        loss = loss_fn(preds,data["label"])
        loss.backward()
        optimizer.step()
        return preds,data["label"]

def predicting(model, device, loader):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data["graphImg"] = data["graphImg"].float().to(device).permute(0, 3, 1, 2)
            data["ecfp"] = data["ecfp"].float().to(device)
            data["hash"] = data["hash"].float().to(device)
            data["list_of_node"] = data["list_of_node"].float().to(device)
            data["list_of_edge"] = data["list_of_edge"].float().to(device)
            data["label"] = data["label"].float().to(device)
            output = model(data["graphImg"],data["ecfp"],data["hash"],
                       data["list_of_node"],data["list_of_edge"],data["list_of_pos"])
            preds = output.cpu()
    return preds,data["label"]

if __name__ == '__main__':
    n_train = len(MCPDataset())
    split = n_train // 5
    indices = np.random.choice(range(n_train), size=n_train, replace=False)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    train_loader = DataLoader(MCPDataset(), sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
    test_loader = DataLoader(MCPDataset(), sampler=test_sampler, batch_size=TEST_BATCH_SIZE)

    # %%
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(NUM_EPOCHS):
        GT, GP = train(model, device, train_loader, optimizer)
        G, P = predicting(model, device, test_loader)
        G, P = G.cpu(),P.cpu()
        # If you want to use regression calculation, please use the following three lines of comments.
        # r2 = r2_score(G, P)
        # rmse = np.sqrt(mean_squared_error(G, P))
        # mae = mean_absolute_error(G, P)
        fpr, tpr, thresholds = roc_curve(P, G)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_new = (G >= optimal_threshold).to(torch.int)
        roc_auc = roc_auc_score(P, y_pred_new)
        ACC = accuracy_score(P, y_pred_new)
        f1 = f1_score(P, y_pred_new)
        print("AUC:",roc_auc,"ACC:",ACC,"f1:",f1)
        # print("r2:", r2,",rmse:",rmse,",mae:",mae)



