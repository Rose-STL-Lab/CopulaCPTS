import os.path
import torch
from torch.utils.data import TensorDataset
import numpy as np


class rnn(torch.nn.Module):
    """
    The auxiliary RNN issuing point predictions.
    Point predictions are used as baseline to which the (normalised)
    uncertainty intervals are added in the main CFRNN network.
    """

    def __init__(
        self, embedding_size=24, input_size=2, output_size=1, horizon=1, path=None
    ):
        """
        Initialises the auxiliary forecaster.
        Args:
            embedding_size: hyperparameter indicating the size of the latent
                RNN embeddings.
            input_size: dimensionality of the input time-series
            output_size: dimensionality of the forecast
            horizon: forecasting horizon
            rnn_mode: type of the underlying RNN network
            path: optional path where to save the auxiliary model to be used
                in the main CFRNN network
        """
        super(rnn, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size
        self.path = path

        # self.forecaster_rnn = torch.nn.LSTM(input_size=input_size, hidden_size=embedding_size, batch_first=True)
        self.forecaster_rnn = torch.nn.RNN(
            input_size=input_size, hidden_size=embedding_size, batch_first=True
        )
        self.forecaster_out = torch.nn.Linear(embedding_size, horizon * output_size)
        self.X = None
        self.y = None

        self.loss_fn = None
        self.loss = None

    def forward(self, x, state=None):

        # [batch, horizon, output_size]
        _, h_n = self.forecaster_rnn(x, state)

        out = self.forecaster_out(h_n).reshape(-1, self.horizon, self.output_size)

        return out, h_n

    def train_model(self, x_train, y_train, n_epochs=100, batch_size=150, lr=0.01):
        """
        Trains the auxiliary forecaster to the training dataset.
        Args:
            x_train, y_train: tensor input
            batch_size: batch size
            epochs: number of training epochs
            lr: learning rate
        """

        self.X = x_train
        self.y = y_train
        print("yshape", y_train.shape)

        train_dataset = TensorDataset(x_train, y_train)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        self.train()
        for epoch in range(n_epochs):
            for sequences, targets in train_loader:

                out, _ = self(sequences.detach())
                self.loss = self.loss_fn(out, targets.detach())

                optimizer.zero_grad()
                self.loss.backward(
                    retain_graph=True
                )  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            # if epoch % 10 == 0:
            #    print("Epoch: ", epoch, "| train loss: %.4f" % self.loss.data)

        if self.path is not None:
            torch.save(self, self.path)

    def predict(self, x):
        """
        x: tensor input
        """

        pred_y, _ = self(x)
        return pred_y
