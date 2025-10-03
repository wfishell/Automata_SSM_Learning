import torch
import torch.nn as nn

from data_prep_ssm import prepare_datasets
from State_Space_Model import FSM_Mamba  # assuming you put the model class there

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load datasets
X_train, Y_train, X_test, Y_test = prepare_datasets(
    "Training_Dataset.txt", ["go", "cancel", "req"], ["grant"]
)

# define model
model = FSM_Mamba(input_dim=3, output_dim=1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(10):
    optimizer.zero_grad()
    y_hat = model(X_train, Y_train, autoregressive=False)
    loss = criterion(y_hat, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: {loss.item():.4f}")

# rollout evaluation
with torch.no_grad():
    y_pred = model(X_test, autoregressive=True)
    print("Inputs:", X_test[0])
    print("Preds :", y_pred[0])
    print("Truth :", Y_test[0])
