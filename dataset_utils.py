# Util functions for dataset operations
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Method shuffles labels at the beginning and store their indexes separately


def split_and_shuffle_labels(y_data, seed, amount):
    y_data = pd.DataFrame(y_data)
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        var_name = "label" + str(i)
        label_info = y_data[y_data["class"] == str(i)]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_dict.update({var_name: label_info})
    return label_dict

# The ındexes that grouped previous method are distributed to node datasets equally


def get_iid_subsamples_indices(label_dict, number_of_nodes, amount):
    sample_dict = dict()

    for i in range(number_of_nodes):
        sample_name = "node"+str(i)
        dumb = pd.DataFrame()
        for j in range(10):
            label_name = str("label")+str(j)
            total = len(label_dict[label_name])
            batch_size = int(total/number_of_nodes)
            a = label_dict[label_name][i*batch_size:(i+1)*batch_size]
            dumb = pd.concat([dumb, a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict

# Creating datasets for each node according to the indexes getting at previous method.
# Images at the related index and labels are transformed to dataset.


def create_iid_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(len(sample_dict)):
        xname = x_name+str(i)
        yname = y_name+str(i)
        node_name = "node"+str(i)

        indices = np.sort(
            np.array(sample_dict[node_name]["i"], dtype=np.int32))

        x_info = np.asfarray(x_data)[indices, :]
        y_info = np.asfarray(y_data)[indices]

        # Preprocessing before going to be dataset.
        x_info, y_info = pre_process_data(x_info, y_info)

        x_data_dict.update({xname: x_info})
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict

    # Images are normalized and labels converted to one hot encoding


def pre_process_data(set_x, set_y):
    # Normalize
    set_x = set_x / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    set_y = enc.fit_transform(set_y.reshape(len(set_y), -1))

    return set_x, set_y
