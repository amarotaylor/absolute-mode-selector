import pickle as pkl
import numpy as np
from keras.layers import Input, Dense, concatenate, Conv1D, Flatten, Reshape
from keras.models import Model

reload_data = False
ABS_features = ['alpha', 'tau', 'b', 'delta', 'LL', 'SCNA_LL', 'Kar_LL']
max_modes_num = 20  # real max is 42

if reload_data:
    file_names = ["pcawg_full_data_basic+_11-35-01.636739.pickle", "TCGA_set_gcloud_basic+_16-36-11.993118.pickle"]

    mut_mat_dict = {}
    modes_dict = {}
    purity_dict = {}
    ploidy_dict = {}
    for fn in file_names:
        with open(fn, "rb") as data_file:
            data_matrix = pkl.load(data_file, encoding='latin1')
            mut_mat_dict.update(data_matrix[0]) # copy number and mutation matrixes per sample (1x300)
            modes_dict.update(data_matrix[1]) # solution modes per samples (correct solutions are marked abs_table_solution_class == 1)
            purity_dict.update(data_matrix[2]) # target purity value
            ploidy_dict.update(data_matrix[3]) # target ploidy value

    mut_mat = np.zeros([len(mut_mat_dict), 301])
    abs_mat = np.zeros([len(modes_dict), len(ABS_features), max_modes_num])
    purity_vec = np.zeros(len(purity_dict))
    ploidy_vec = np.zeros(len(ploidy_dict))
    for i, key in enumerate(mut_mat_dict.keys()):
        mut_mat[i, :] = mut_mat_dict[key]
        tmp = modes_dict[key]
        purity_vec[i] = purity_dict[key]
        ploidy_vec[i] = ploidy_dict[key]
        modes_shape = modes_dict[key].shape
        if modes_shape[0] > max_modes_num:
            abs_mat[i, :, :] = np.transpose(modes_dict[key].loc[:max_modes_num - 1, ABS_features].values)
        else:
            abs_mat[i, :, :modes_shape[0]] = np.transpose(modes_dict[key].loc[:, ABS_features].values)

    np.save('mutations matrix', mut_mat)
    np.save('ABSOLUTE matrix', abs_mat)
    np.save('purity matrix', purity_vec)
    np.save('ploidy matrix', ploidy_vec)
else:
    mut_mat = np.load('mutations matrix.npy')
    abs_mat = np.load('ABSOLUTE matrix.npy')
    purity_vec = np.load('purity matrix.npy')
    ploidy_vec = np.load('ploidy matrix.npy')



# Initialize input layers
mut_input = Input(shape=(301,))
abs_input = Input(shape=(len(ABS_features), max_modes_num))

# Build the purity network
pur_dense1 = Dense(200, activation='softsign', kernel_initializer='glorot_normal')(mut_input)
pur_dense2 = Dense(100, activation='softsign', kernel_initializer='glorot_normal')(pur_dense1)
pur_dense3 = Dense(50, activation='softsign', kernel_initializer='glorot_normal')(pur_dense2)
predictions = Dense(1, activation='softsign')(pur_dense3)

#abs_conv = Conv1D(16, 7)(abs_input)
abs_dense1 = Dense(100, activation='softsign', kernel_initializer='glorot_normal')(abs_input)
abs_dense2 = Dense(50, activation='softsign', kernel_initializer='glorot_normal')(abs_dense1)
abs_flat = Flatten()(abs_dense2)
merged = concatenate([abs_flat, pur_dense3])

predictions = Dense(1, activation='softsign')(merged)

model = Model(inputs=[abs_input, mut_input], outputs=predictions)
#model = Model(inputs=mut_input, outputs=predictions)
model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit([abs_mat, mut_mat], purity_vec, epochs=7, batch_size=100, verbose=2)
#model.fit(mut_mat, purity_vec, epochs=7, batch_size=100, verbose=2)