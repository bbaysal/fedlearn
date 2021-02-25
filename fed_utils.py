import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Util functions for federated learning operations
def get_avaraged_weights(model_dict, number_of_nodes, len_layers):
    avarages_dict = dict()

    # Loop over number of layer in network
    # For the first step, zero-fill matrix are created only. 
    for i in range(1, len_layers):
        model = "model0"
        weight_name = "W"+str(i)
        bias_name = "b"+str(i)
        weight = np.zeros(model_dict[model].get_parameter(weight_name).shape)
        bias = np.zeros(model_dict[model].get_parameter(bias_name).shape)
        avarages_dict.update({weight_name: weight})
        avarages_dict.update({bias_name: bias})

    # Loop over all nodes and alsı all layers. 
    for i in range(number_of_nodes):
        model = "model" + str(i)
        for j in range(1, len_layers):
            weight_name = "W"+str(j)
            bias_name = "b"+str(j)

            weight = avarages_dict[weight_name] + \
                model_dict[model].get_parameter(weight_name)
            bias = avarages_dict[bias_name] + \
                model_dict[model].get_parameter(bias_name)

            avarages_dict.update({weight_name: weight})
            avarages_dict.update({bias_name: bias})

    # Averaging weights and biases at the end.
    for i in range(1, len_layers):
        weight_name = "W"+str(i)
        bias_name = "b"+str(i)
        weight = avarages_dict[weight_name] / number_of_nodes
        bias = avarages_dict[bias_name] / number_of_nodes
        avarages_dict.update({weight_name: weight})
        avarages_dict.update({bias_name: bias})

    return avarages_dict

# Updating main model parameters according to average weights of nodes. 
def set_main_model_weights(main_model, model_dict, len_layers):
    avg_dict = get_avaraged_weights(model_dict, len(model_dict))

    for i in range(1, len_layers):
        weight_name = "W"+str(i)
        bias_name = "b"+str(i)
        main_model.set_parameter(weight_name, avg_dict[weight_name])
        main_model.set_parameter(bias_name, avg_dict[bias_name])

    return main_model

# Send main model parameters to all nodes. 
def set_node_weights(main_model, model_dict, len_layers):
    for i in range(len(model_dict)):
        model_name = "model" + str(i)
        for j in range(1, len_layers):
            weight_name = "W"+str(j)
            bias_name = "b"+str(j)
            model_dict[model_name].set_parameter(
                weight_name, main_model.get_parameter(weight_name))
            model_dict[model_name].set_parameter(
                bias_name, main_model.get_parameter(bias_name))
    return model_dict

def test_nodes_and_main_model(main_model, model_dict, test_x_dict, test_y_dict):
    nodes = len(model_dict)
    table = pd.DataFrame(np.zeros([nodes, 3]), columns=[
        "node", "node_model", "main_model"])
    for i in range(nodes):
        test_input_name = "test_x" + str(i)
        test_target_name = "test_y" + str(i)
        X = test_x_dict[test_input_name]
        y = test_y_dict[test_target_name]
        model_name = "model" + str(i)
        model = model_dict[model_name]

        acc = model.predict(X, y)
        main_acc = main_model.predict(X, y)

        table.loc[i, "node"] = "Node "+str(i)
        table.loc[i, "node_model"] = acc
        table.loc[i, "main_model"] = main_acc

    return table