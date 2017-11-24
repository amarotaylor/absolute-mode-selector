import pickle as pkl
import numpy as np
from types import SimpleNamespace
from random import shuffle
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, concatenate, Conv1D, Flatten, Reshape
from keras.models import Model

reload_data = False
plot_results = False
# ABS_features = ['alpha', 'tau', 'b', 'delta', 'LL', 'SCNA_LL', 'Kar_LL']
max_modes_num = 20  # real max is 42
test_size = 1000


def createNumpyDataSet(data, keys, mode_num):
    np_data = SimpleNamespace()
    sample_num = len(keys)
    np_data.mut_mat = np.zeros([sample_num, 301])
    np_data.abs_mat = np.zeros([sample_num, len(data.modes_dict[list(keys)[0]].columns) - 1, mode_num])
    np_data.purity_vec = np.zeros(sample_num)
    np_data.ploidy_vec = np.zeros(sample_num)
    np_data.solution_idx = np.zeros(sample_num)
    for i, key in enumerate(keys):
        np_data.mut_mat[i, :] = data.mut_mat_dict[key]
        np_data.purity_vec[i] = data.purity_dict[key]
        np_data.ploidy_vec[i] = data.ploidy_dict[key]
        np_data.solution_idx[i] = data.modes_dict[key].index[data.modes_dict[key]['solution_class'] == 1][0]
        modes_shape = data.modes_dict[key].shape
        data.modes_dict[key].drop('solution_class', axis=1, inplace=True)  # remove the correct mode indicator from input
        if modes_shape[0] > mode_num:  # truncate the modes according to predefined value
            np_data.abs_mat[i, :, :] = np.nan_to_num(np.transpose(data.modes_dict[key].loc[:mode_num - 1, :].values))
        else:
            np_data.abs_mat[i, :, :modes_shape[0]] = np.nan_to_num(np.transpose(data.modes_dict[key].loc[:, :].values))
    return np_data


if reload_data:
    file_names = ["pcawg_full_data_basic+_11-35-01.636739.pickle", "TCGA_set_gcloud_basic+_16-36-11.993118.pickle"]

    all_data = SimpleNamespace()
    all_data.mut_mat_dict = {}
    all_data.modes_dict = {}
    all_data.purity_dict = {}
    all_data.ploidy_dict = {}

    print('Loading data from the following files: {}'.format(file_names))
    for fn in file_names:
        with open(fn, "rb") as data_file:
            data_matrix = pkl.load(data_file, encoding='latin1')
            all_data.mut_mat_dict.update(data_matrix[0]) # copy number and mutation matrixes per sample (1x300)
            all_data.modes_dict.update(data_matrix[1]) # solution modes per samples (correct solutions are marked abs_table_solution_class == 1)
            all_data.purity_dict.update(data_matrix[2]) # target purity value
            all_data.ploidy_dict.update(data_matrix[3]) # target ploidy value

    print('Dividing train and test set...')
    all_keys = list(all_data.ploidy_dict.keys())
    shuffle(all_keys)
    test_keys = all_keys[:test_size]
    train_keys = all_keys[test_size:]

    test_set = createNumpyDataSet(all_data, test_keys, max_modes_num)
    train_set = createNumpyDataSet(all_data, train_keys, max_modes_num)

    with open(r"Test Set.pickle", "wb") as test_file:
        pkl.dump(test_set, test_file)
    with open(r"Train Set.pickle", "wb") as train_file:
        pkl.dump(train_set, train_file)
else:
    print('Loading the train and test sets...')
    with open(r"Test Set.pickle", "rb") as test_file:
        test_set = pkl.load(test_file)
    with open(r"Train Set.pickle", "rb") as train_file:
        train_set = pkl.load(train_file)

print('Initializing the network...')
# Initialize input layers
activation = 'sigmoid'
initialization = 'glorot_normal'
mut_input = Input(shape=(301,))
abs_input = Input(shape=(23, max_modes_num))

# Build the purity network
pur_dense1 = Dense(256, activation=activation, kernel_initializer=initialization)(mut_input)
pur_dense2 = Dense(128, activation=activation, kernel_initializer=initialization)(pur_dense1)
pur_dense3 = Dense(64, activation=activation, kernel_initializer=initialization)(pur_dense2)

abs_dense1 = Dense(128, activation=activation, kernel_initializer=initialization)(abs_input)
abs_dense2 = Dense(64, activation=activation, kernel_initializer=initialization)(abs_dense1)
abs_flat = Flatten()(abs_dense2)
merged = concatenate([abs_flat, pur_dense3])

predictions = Dense(1, activation=activation)(merged)

model = Model(inputs=[abs_input, mut_input], outputs=predictions)
print('Compiling...')
model.compile(optimizer='Adamax', loss='mean_squared_error')
print('Training...')
model.fit([train_set.abs_mat, train_set.mut_mat], train_set.purity_vec, epochs=10, batch_size=128, verbose=2)
y_pred = model.predict([test_set.abs_mat, test_set.mut_mat])
score = model.evaluate([test_set.abs_mat, test_set.mut_mat], test_set.purity_vec)
print('Test set loss is: {}'.format(score))
#print('Pearson correlation between predicted and real values: {}'.format(np.corrcoef(test_set.purity_vec, y_pred)))
if plot_results:
    plt.scatter(test_set.purity_vec, y_pred)
    plt.show()

success_num = 0
test_sample_num = len(test_set.solution_idx)
for i in range(test_sample_num):
    abs_purity = list(test_set.abs_mat[i, 0, :])  # get all possible purity values from the ABSOLUTE's output
    pred_tuple = min(enumerate(abs_purity), key=lambda x: abs(x[1] - y_pred[i]))
    real_idx = test_set.solution_idx[i]
    #print(real_idx, pred_tuple, y_pred[i], abs_purity)
    if real_idx == pred_tuple[0]:
        success_num += 1
        print(real_idx, pred_tuple[0])
pred_accuracy = success_num / test_sample_num

print('Modes accuracy is: {}'.format(pred_accuracy))

print('Done!')
