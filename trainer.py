import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from backend.complexmodel import *
import torch.nn.functional as F
from backend.arguments import *
import sys
args=sys.argv
keys=[('-i', "datasets/output.csv"), ('-o', "models/model.pth"), ('-e', "200000"), ('-lr', "0.000001")]
values = extract_values_with_defaults(keys=keys, argv=args)

dataset=values['-i']
output=values['-o']
num_epochs=int(values['-e'])
lr=float(values['-lr'])

print(f"Beginning training with data from {dataset} and {str(num_epochs)} epochs, with a learning rate of {str(lr)}\nPlease note: this file does not call model.eval()")

data = pd.read_csv(dataset)

# Assuming the first column is the target and the rest are features
# Assuming the last two columns 'MX' and 'MY' are the targets
X = data.iloc[:, :-2].values  # Selecting all columns except the last two
y = data[['D']].values  # Selecting only the 'MX' and 'MY' columns


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

input_size = X_train.shape[1]
output_size = 1
hidden_size = 10
num_hidden_layers = 10

model = ComplexModel(input_size, output_size, hidden_size, num_hidden_layers)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)

x_values=[]
y_values=[]
lossP=float("inf")
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(y_train_tensor, outputs)
    x_values.append(epoch)
    y_values.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % (num_epochs//10) == 0:
        x_values=[]
        y_values=[]
        if(loss.item()==lossP):
            print(outputs)

        if(loss<lossP):
            lossP=loss.item()
            torch.save(model.state_dict(), f"bestmodel")
            print("saving model")
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')