import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def training_lr(model, x, y, learning_rate=0.001, batch_size=50, num_epoch=1000):
    """
    Train a Bayesian linear regression model with stochastic variational inference on data (x, y)
    """
    parameters = set(model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-3)
    criterion = nn.MSELoss()

    train_errors = []

    num_data, _ = x.shape
    _, num_output = y.shape
    data = torch.cat((x, y), 1)

    for epoch in range(num_epoch):
        # permuate the data
        data_perm = data[torch.randperm(len(data))]
        x = data_perm[:, :-num_output]
        y = data_perm[:, -num_output:]
        for index in range(int(num_data / batch_size)):
            # data comes in
            inputs = x[index * batch_size: (index + 1) * batch_size]
            labels = y[index * batch_size: (index + 1) * batch_size]
            # initialize the gradient of optimizer
            optimizer.zero_grad()
            model.train()
            output, kl = model(inputs)
            # calculate the training loss
            loss = criterion(labels, output) / 2 + kl / num_data
            # backpropogate the gradient
            loss.backward()
            # optimize with SGD
            optimizer.step()
        # validation loss
        model.eval()

        # splite the training data
        output_x_train, kl = model(x)

        train_errors.append(criterion(output_x_train, y).detach())
        if ((epoch + 1) % 4000) == 0:
            print('EPOACH %d: TRAIN LOSS: %.4f; KL REG: %.4f.' % (epoch + 1, train_errors[epoch], kl))


def training_hs_lr(model, x, y, learning_rate=0.001, batch_size=50, num_epoch=1000):
    """
    Train a Bayesian linear regression model with the horseshoe prior with stochastic variational inference on data (x, y)
    """
    parameters = set(model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-3)
    criterion = nn.MSELoss()

    train_errors = []

    num_data, _ = x.shape
    _, num_output = y.shape
    data = torch.cat((x, y), 1)

    for epoch in range(num_epoch):
        # permuate the data
        data_perm = data[torch.randperm(len(data))]
        x = data_perm[:, :-num_output]
        y = data_perm[:, -num_output:]
        for index in range(int(num_data / batch_size)):
            # data comes in
            inputs = x[index * batch_size: (index + 1) * batch_size]
            labels = y[index * batch_size: (index + 1) * batch_size]
            # initialize the gradient of optimizer
            optimizer.zero_grad()
            model.train()
            output, kl = model(inputs)
            # calculate the training loss
            loss = criterion(labels, output) / 2 + kl / num_data
            # backpropogate the gradient
            loss.backward()
            # optimize with SGD
            optimizer.step()
            # analytical update
            model._updates()
        # validation loss
        model.eval()

        # splite the training data
        output_x_train, kl = model(x)

        train_errors.append(criterion(output_x_train, y).detach())
        if ((epoch + 1) % 4000) == 0:
            print('EPOACH %d: TRAIN LOSS: %.4f; KL REG: %.4f.' % (epoch + 1, train_errors[epoch], kl))


def training_nn(model, x, y, x_test, y_test, learning_rate=0.001, batch_size=50, num_epoch=1000):
    parameters = set(model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate, eps=1e-3)
    criterion = nn.MSELoss()

    train_errors = []
    test_errors = []

    num_data, _ = x.shape
    _, num_output = y.shape
    data = torch.cat((x, y), 1)

    for epoch in range(num_epoch):
        # permuate the data
        data_perm = data[torch.randperm(len(data))]
        x = data_perm[:, :-num_output]
        y = data_perm[:, -num_output:]
        for index in range(int(num_data / batch_size)):
            # data comes in
            inputs = x[index * batch_size: (index + 1) * batch_size]
            labels = y[index * batch_size: (index + 1) * batch_size]
            # initialize the gradient of optimizer
            optimizer.zero_grad()
            model.train()
            output, kl, sigma_n = model(inputs)
            # calculate the training loss
            loss = criterion(labels, output) / (2 * sigma_n ** 2) + torch.log(
                sigma_n * np.sqrt(2. * np.pi)) + kl / num_data
            # backpropogate the gradient
            loss.backward()
            # optimize with SGD
            optimizer.step()
        # validation loss
        model.eval()

        # splite the training data
        output_x_train, kl, _ = model(x)
        output_x_test, kl, _ = model(x_test)

        train_errors.append(criterion(output_x_train, y).detach())
        test_errors.append(criterion(output_x_test, y_test).detach())
        if (epoch % 500) == 0:
            print('EPOACH %d: TRAIN LOSS: %.4f; KL REG: %.4f; TEST LOSS IS: %.5f.' % (
            epoch + 1, train_errors[epoch], kl, test_errors[epoch]))

    return train_errors, test_errors