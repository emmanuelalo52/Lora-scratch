import torch
import torch.nn as nn
import torch.functional as F
x = torch.rand((1000,20))

y = (torch.sin(x.sum(1))>0).long()

unique, counts = torch.unique(y, return_counts=True)
distribution = dict(zip(unique.tolist(), counts.tolist()))

n_train = 800
batch_size = 64

#dataloader
train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:n_train],y[:n_train]),batch_size=batch_size,shuffle=True,)

eval_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[n_train:],y[n_train:]),batch_size=batch_size)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20,2000),
            nn.ReLU(),
            nn.Linear(2000,200),
            nn.ReLU(),
            nn.Linear(200,2),
            nn.LogSoftmax(dim=-1),
        )
    def forward(self,x):
        return self.seq(x)
    
lr = 0.002
batch_size = 64
max_epochs = 35
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
print(device)


# Data and Model have to put on the device.
# So, xb, yb and model are on device.
def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        for xb, yb in eval_dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                outputs = model(xb)
            loss = criterion(outputs, yb)
            eval_loss += loss.detach().float()

        eval_loss_total = (eval_loss / len(eval_dataloader)).item()
        train_loss_total = (train_loss / len(train_dataloader)).item()
        print(f"{epoch=:<2}  {train_loss_total=:.4f}  {eval_loss_total=:.4f}")

base_model = MLP().to(device)
optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

#train params
def trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _,params in model.named_parameters():
        all_params+=params.numel()
        if params.requires_grad:
            trainable_params+=params.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

# print(trainable_parameters(base_model))

print(train(base_model, optimizer, criterion, train_dataloader, eval_dataloader, epochs=20))