import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pylab as plt
from nn import NN
from dataset_utils import *
from fed_utils import *

# number_of_nodes specifies number of edges in the network.
number_of_nodes = 10

# Traditional NN layer dimensions. First value should be equal to input dimensions.
layers_dims = [28*28, 200,  10]
len_layers = len(layers_dims)
learning_rate = 0.1
epoch = 1000


def create_nodes(number_of_nodes):
    model_dict = dict()

    for i in range(number_of_nodes):
        model_name = "model"+str(i)
        model_info = NN(layers_dims)
        model_dict.update({model_name: model_info})

    return model_dict

# Training all nodes.


def train_nodes(model_dict, train_x_dict, train_y_dict):
    model_costs = dict()
    for i in range(len(model_dict)):
        model_name = "model"+str(i)
        train_input_name = "train_x" + str(i)
        train_target_name = "train_y" + str(i)
        X = train_x_dict[train_input_name]
        y = train_y_dict[train_target_name]

        model = model_dict[model_name]
        model.fit(X, y, learning_rate=learning_rate,
                  n_iterations=epoch, model_name=model_name)
        model_costs.update({model_name: model.costs})
        model_dict.update({model_name: model})
    return model_costs


X, y = fetch_openml('mnist_784', version=1,
                    return_X_y=True, data_home="./data/")
print(X)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1)
print(train_x)

# Prepearing raw dataset to federated learning.
label_dict = split_and_shuffle_labels(train_y, 1, train_y.shape[0])
sample_dict = get_iid_subsamples_indices(
    label_dict, number_of_nodes,  train_y.shape[0])
train_x_dict, train_y_dict = create_iid_subsamples(
    sample_dict, train_x, train_y, "train_x", "train_y")

label_dict = split_and_shuffle_labels(test_y, 1, test_y.shape[0])
sample_dict = get_iid_subsamples_indices(
    label_dict, number_of_nodes,  test_y.shape[0])
test_x_dict, test_y_dict = create_iid_subsamples(
    sample_dict, test_x, test_y, "test_x", "test_y")

# Main model the netowrk.
main_model = NN(layers_dims)

# Nodes will be created
model_dict = create_nodes(number_of_nodes)

# Accuracy values before training

table = test_nodes_and_main_model(
    main_model, model_dict, test_x_dict, test_y_dict)
print(table)

# Setting node weights same as main model
set_node_weights(main_model, model_dict, len_layers)

# Training
model_costs = train_nodes(model_dict, train_x_dict, train_y_dict)

# Adjusting  main model weights  with average value
set_main_model_weights(main_model, model_dict)

# Accuract values after training. Now model and node values are differeny.
table = test_nodes_and_main_model(
    main_model, model_dict, test_x_dict, test_y_dict)
print(table)

# setting average value all throughtout network
set_node_weights(main_model, model_dict, len_layers)

# Accuract values after setting weights to average
table = test_nodes_and_main_model(
    main_model, model_dict, test_x_dict, test_y_dict)
print(table)

n_cols=2
n_rows=int(number_of_nodes/n_cols)
# Creating 2x2 plot area
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12,12))
plt.subplots_adjust(wspace=0.3, hspace=0.7)
for ax,(name, value) in zip(axes.reshape(-1), model_costs.items()):
  ax.plot(value)
  ax.set(xlabel='Epoch', ylabel='Loss', title=name)


plt.show()