This folder contains data from a proof-of-concept test.
Before training EtaZero, we should test that the graph neural network can train successfully.

data.csv contains the training data:
    - X = game states from many games between two UCT agents
    - y = whether the current player won the game

Only data with num tiles < 10 were used since the winner between two mediocre agents at the start is unpredictable.
Accuracy is calculated as binary classification where the single output is rounded to 1 or -1.
90% accuracy on unseen data achieved!

training data size:     977
validation data size:   244

train_history contains 3 columns:
    - epoch number
    - MSE loss
    - accuracy on validation data set

NN hyperparameters:
    - dims = [3,10,10,1]
    - aggregator = max
    - activations = relu, tanh last
    - batch size = 32
    - epochs = 20
    - optimizer = Adam
    - learning rate = 0.001
