# https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVR
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class FraudData(Dataset):
    def __init__(self, train=True):
        df = pd.read_csv("onlinefraud.csv")
        df = df.sample(n=10000).reset_index(drop=True) # Randomly sample 10,000 rows out of 6.36m
        df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True) # drops id rows of buyer and seller

        # Uncomment if you'd like to display the string values of categorical data
        #
        # # Create a dictionary for category columns
        # cat_columns = df.select_dtypes("object").columns  # Select all columns of tye object (catergory columns)
        # df[cat_columns] = df[cat_columns].astype("category")  # cast them to type category
        # # Make a dictionary of the different categories converted to numbers
        # cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
        #                              range(len(df[cat_columns[i]].cat.categories))} for i
        #             in range(len(cat_columns))}
        # print(cat_dict)  # Print the dictionary for deciphering the category numbers

        # Replace categories with int values
        df[df.select_dtypes("category").columns] = df[df.select_dtypes("category").columns].apply(lambda x: x.cat.codes)

        # Feature and label data
        X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32).view(-1, 7)
        y = torch.tensor(df['isFraud'], dtype=torch.float32)

        # Split data into train and test with train being 70% of data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        if train:
            self.X = X_train.clone().detach().float()
            self.y = y_train.clone().detach().float()
        else:
            self.X = X_test.clone().detach().float()
            self.y = y_test.clone().detach().float()

        # Determine the length of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X), np.array(self.y)


class FraudFit(nn.Module):
    def __init__(self):
        super(FraudFit, self).__init__()

        # Uses only 1 hidden layer
        self.in_to_h1 = nn.Linear(7, 4)
        self.h1_to_out = nn.Linear(4, 1)

    def forward(self, x):
        # applies sigmoid function on hidden layer
        x = F.sigmoid(self.in_to_h1(x))
        x = self.h1_to_out(x)
        return x


def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=25):
    cd = FraudData()

    # Create a data loader that shuffles each time and allows for the last batch to be smaller
    # than the rest of the epoch if the batch size doesn't divide the training set size
    curve_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create neural network
    fraud_network = FraudFit()

    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')

    # Select the optimizer
    optimizer = torch.optim.Adam(fraud_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(curve_loader, 0):
            x, y = data

            # resets gradients to zero
            optimizer.zero_grad()

            # evaluate the neural network on x
            output = fraud_network(x)

            # compare to the actual label value
            loss = mse_loss(output.view(-1, 1), y.view(-1, 1))

            # perform back propagation
            loss.backward()

            # perform gradient descent with an Adam optimizer
            optimizer.step()

            # update the total loss
            running_loss += loss.item()

        # every epoch_display epochs give the mean square error since the last update
        # this is averaged over multiple epochs
        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(curve_loader) * epoch_display):.6f}")
            running_loss = 0.0
    return fraud_network, cd


# trains network with number of epochs passed in as the parameter
cn, cd = trainNN(epochs=100)

X_numpy, y_numpy = cd.to_numpy()

# Runs network on test data and prints the Mean Square Error
cd_test = FraudData(train=False)
with torch.no_grad():
    y_pred_test = cn(cd_test.X).view(-1)
    test_mse = np.average((cd_test.y.numpy() - y_pred_test.numpy()) ** 2)
    print(f"Test MSE: {test_mse}")