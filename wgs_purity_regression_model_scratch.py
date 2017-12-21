from argparse import Namespace
import pickle as pkl
import numpy as np
from argparse import Namespace
from random import shuffle
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, concatenate, Conv1D, Flatten, Reshape
from keras.models import Model, Sequential
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import copy

def createNumpyDataSet(data, keys, mode_num):
    np_data = Namespace()
    sample_num = len(keys)
    np_data.mut_mat = np.zeros([sample_num, 301])
    np_data.abs_mat = np.zeros([sample_num, len(data.modes_dict[list(keys)[0]].columns) - 1, mode_num])
    np_data.purity_vec = np.zeros(sample_num)
    np_data.ploidy_vec = np.zeros(sample_num)
    np_data.wgd_vec = np.zeros(sample_num)
    np_data.solution_idx = np.zeros(sample_num)
    for i, key in enumerate(keys):
        np_data.mut_mat[i, :] = data.mut_mat_dict[key]
        np_data.purity_vec[i] = data.purity_dict[key]
        np_data.ploidy_vec[i] = data.ploidy_dict[key]
        np_data.solution_idx[i] = data.modes_dict[key].index[data.modes_dict[key]['solution_class'] == 1][0]
        np_data.wgd_vec[i] = data.modes_dict[key]['WGD'][np_data.solution_idx[i]]
        data.modes_dict[key].drop('solution_class', axis=1,
                                  inplace=True)  # remove the correct mode indicator from input
        modes_shape = data.modes_dict[key].shape
        if modes_shape[0] > mode_num:  # truncate the modes according to predefined value
            np_data.abs_mat[i, :, :] = np.nan_to_num(np.transpose(data.modes_dict[key].loc[:mode_num - 1, :].values))
        else:
            np_data.abs_mat[i, :, :modes_shape[0]] = np.nan_to_num(np.transpose(data.modes_dict[key].loc[:, :].values))
    return np_data


file_names = ["/home/amaro/Projects/AbsSolCaller/pcawg_full_data_basic+_11-350-01.636739.pickle"
    , "/home/amaro/Projects/AbsSolCaller/TCGA_set_gcloud_basic+_16-36-11.993118.pickle"]
all_data = Namespace()
all_data.mut_mat_dict = {}
all_data.modes_dict = {}
all_data.purity_dict = {}
all_data.ploidy_dict = {}
for fn in file_names:
    with open(fn, "rb") as data_file:
        data_matrix = pkl.load(data_file)
        all_data.mut_mat_dict.update(data_matrix[0])  # copy number and mutation matrixes per sample (1x300)
        all_data.modes_dict.update(data_matrix[1])
        all_data.purity_dict.update(data_matrix[2])  # target purity value
        all_data.ploidy_dict.update(data_matrix[3])  # target ploidy value

print('Dividing train and test set')
test_size = 1000
max_modes_num = 20

all_keys = list(all_data.ploidy_dict.keys())
shuffle(all_keys)
test_keys = all_keys[:test_size]
test_keys = all_keys[:test_size]
train_keys = all_keys[test_size:]

test_set = createNumpyDataSet(all_data, test_keys, max_modes_num)
train_set = createNumpyDataSet(all_data, train_keys, max_modes_num)
with open(r"Test_SetpickleWGD", "wb") as test_file:
    pkl.dump(test_set, test_file)

with open(r"Train_SetpickleWGD", "wb") as train_file:
    pkl.dump(train_set, train_file)

print('Initializing the network')
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
merged_purity = concatenate([abs_flat, pur_dense3])
predictions_purity = Dense(1, activation=activation)(pur_dense3)

# Build the ploidy network
ploidy_dense1 = Dense(256, activation=activation, kernel_initializer=initialization)(mut_input)
ploidy_dense2 = Dense(128, activation=activation, kernel_initializer=initialization)(ploidy_dense1)
ploidy_dense3 = Dense(64, activation=activation, kernel_initializer=initialization)(ploidy_dense1)

# Build the ploidy network
ploidy_dense1 = Dense(256, activation=activation, kernel_initializer=initialization)(mut_input)
ploidy_dense2 = Dense(128, activation=activation, kernel_initializer=initialization)(ploidy_dense1)
ploidy_dense3 = Dense(64, activation=activation, kernel_initializer=initialization)(ploidy_dense1)

WGD_dense1 = Dense(256, activation=activation, kernel_initializer=initialization)(mut_input)
WGD_dense2 = Dense(128, activation=activation, kernel_initializer=initialization)(WGD_dense1)
WGD_dense3 = Dense(64, activation=activation, kernel_initializer=initialization)(WGD_dense2)
predictions_wgd = Dense(1, kernel_initializer='normal', activation='sigmoid')(WGD_dense3)
model_wgd = Model(inputs=[mut_input], outputs=predictions_wgd)
model_wgd.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_wgd.fit([train_set.mut_mat], train_set.wgd_vec, epochs=10, batch_size=128, verbose=2)


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(301, input_dim=301, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

X = train_set.mut_mat

# encode class values as integers
encoder = LabelEncoder()

encoder.fit(train_set.wgd_vec)
# fix as a ploidy switch removing info about multiple wgds
train_set.wgd_vec[train_set.wgd_vec > 1] = 1
encoded_Y = encoder.transform(train_set.wgd_vec)

seed = 7
np.random.seed(seed)
def wgd_model():
    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=301, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=wgd_model, epochs=200, batch_size=10, verbose=1)))
pipeline_wgd = Pipeline(estimators)
pipeline_wgd.fit(X,encoded_Y)

predict_wgd_test = pipeline_wgd.predict(test_set.mut_mat)
test_set.wgd_vec[test_set.wgd_vec >1 ] =1
np.sum(np.squeeze(predict_wgd_test) - np.squeeze(test_set.wgd_vec) == 0)
#922

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
#print("wgd_model: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, train_set.mut_mat, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

abs_flat = Flatten()(abs_dense2)
merged_ploidy = concatenate([abs_flat, ploidy_dense3])
predictions_ploidy = Dense(1, activation='hard_sigmoid')(ploidy_dense3)

model_purity = Model(inputs=[mut_input], outputs=predictions_purity)
print('Compiling')

model_purity.compile(optimizer='Adamax', loss='mean_squared_error')
print('Training')

model_ploidy = Model(inputs=[mut_input], outputs=predictions_ploidy)
print('Compiling')

model_ploidy.compile(optimizer='Adamax', loss='mean_squared_error')
print('Training')

model_ploidy.fit([train_set.mut_mat], train_set.ploidy_vec, epochs=10, batch_size=128, verbose=2)



model_purity.fit([train_set.mut_mat], train_set.purity_vec, epochs=10, batch_size=128, verbose=2)
train_purity_predictions = model_purity.predict(train_set.mut_mat)
train_wgd_predictions = pipeline.predict(train_set.mut_mat)
train_keys

def createModeClassificationDataSet(data,purity_pred,wgd_pred, keys):
    #np_data = Namespace()
    #sample_num = len(keys)
    #np_data.mut_mat = np.zeros([sample_num, 301])
    #np_data.abs_mat = np.zeros([sample_num, len(data.modes_dict[list(keys)[0]].columns) - 1, mode_num])
    #np_data.purity_vec = np.zeros(sample_num)
    #np_data.wgd_vec = np.zeros(sample_num)
    #np_data.solution_idx = np.zeros(sample_num)
    for i, key in enumerate(keys):
        x_new = data.modes_dict[key]
        pred_purity = np.zeros(len(x_new)) + purity_pred[i]
        pred_wgd = np.zeros(len(x_new)) + wgd_pred[i]
        x_new['pred_purity'] = pred_purity
        x_new['pred_wgd'] = pred_wgd
        x_new['purity_diff'] = np.abs(x_new['pred_purity'] - x_new['alpha'])
        if i == 0:
            X_modes = x_new
        else:
            X_modes = pd.concat([X_modes,x_new])
    X_modes.dropna(axis=1, how='all', inplace=True)
    Y = X_modes['solution_class']
    Y_array = Y.as_matrix()
    X_modes.drop('solution_class', axis=1,inplace=True)
    X_array = X_modes.reset_index().values
    return X_array, Y_array

[X_array,Y_array] = createModeClassificationDataSet(all_data,train_purity_predictions,train_wgd_predictions, train_keys)
# encode class values as integers
encoder = LabelEncoder()

encoder.fit(Y_array)
# fix as a ploidy switch removing info about multiple wgds
solution_class = encoder.transform(Y_array)
def solution_classifier():
    # create model
    model = Sequential()
    model.add(Dense(23, input_dim=23, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=solution_classifier, epochs=200, batch_size=10, verbose=1)))
pipeline_solutions = Pipeline(estimators)
pipeline_solutions.fit(X,solution_class)


test_purity_predictions = model_purity.predict(test_set.mut_mat)
test_wgd_predictions = pipeline_wgd.predict(test_set.mut_mat)

def EvaluateSolutionsPicker(data,purity_pred,wgd_pred, keys):
    picked_correct = 0
    did_not_pick = 0
    picked_wrong = 0
    wrong_idxs = list()
    solutions_picked = list()
    for i, key in enumerate(keys):
        x_test = copy.deepcopy(data.modes_dict[key])
        x_test['pred_purity'] = np.zeros(len(x_test))+purity_pred[i]
        x_test['pred_wgd'] = np.zeros(len(x_test))+wgd_pred[i]
        x_test['purity_diff'] = np.abs(x_test['pred_purity'] - x_test['alpha'])
        x_test.dropna(axis=1,how='all',inplace=True)
        solution_test = x_test['solution_class'].as_matrix()
        x_test.drop('solution_class',axis=1,inplace=True)
        x_test_array = x_test.reset_index().values
        solution_array = pipeline_solutions.predict(x_test_array)
        true_vals = [i for i, x in enumerate(solution_array) if x]
        if np.sum(solution_test[true_vals]) > 0:
            picked_correct = picked_correct+1
            solutions_picked.append(np.max(true_vals)+1)
        elif np.sum(solution_array) == 0:
            did_not_pick = did_not_pick+1
        elif np.sum(solution_array) > 0:
            wrong_idxs.append(key)
            picked_wrong = picked_wrong + 1
    print 'returning'
    print wrong_idxs
    return np.true_divide(picked_correct,len(purity_pred)), np.true_divide(did_not_pick,len(purity_pred)), np.true_divide(picked_wrong,len(purity_pred)),solutions_picked


[correct_frac,did_not_pick_frac,picked_wrong_frac,solutions_picked] = EvaluateSolutionsPicker(all_data,test_purity_predictions,test_wgd_predictions,test_keys)


def manual_review(key,test_keys):
    i= test_keys.index(key)
    x_test = copy.deepcopy(all_data.modes_dict[key])
    x_test['pred_purity'] = np.zeros(len(x_test))+test_purity_predictions[i]
    x_test['pred_wgd'] = np.zeros(len(x_test))+test_wgd_predictions[i]
    x_test['purity_diff'] = np.abs(x_test['pred_purity'] - x_test['alpha'])
    x_test.dropna(axis=1,how='all',inplace=True)
    solution_test = x_test['solution_class'].as_matrix()
    #x_test.drop('solution_class',axis=1,inplace=True)
    x_test_array = x_test.drop('solution_class',axis=1).reset_index().values
    solution_array = pipeline_solutions.predict(x_test_array)
    true_vals = [i for i, x in enumerate(solution_array) if x]
    print x_test
    print solution_array


key = 'CESC-C5-A1MJ-TP-NB'
manual_review(key, test_keys)

picked_correct = 0
did_not_pick = 0
wrong_idxs = list()
if np.sum(solution_test[true_vals]) > 0:
    picked_correct = picked_correct + 1
elif np.sum(solution_test) == 0:
    did_not_pick = did_not_pick + 1
elif np.sum(solution_test) > 0:
    wrong_idxs.append(key)


model_json = pipeline_wgd.named_steps['mlp'].model.to_json()
with open("model_wgd.json", "w") as json_file:
 json_file.write(model_json)

pipeline_wgd.named_steps['mlp'].model.save_weights('wgd_weights.h5')

model_json = pipeline_solutions.named_steps['mlp'].model.to_json()
with open("model_solutions.json", "w") as json_file:
 json_file.write(model_json)

pipeline_solutions.named_steps['mlp'].model.save_weights('solutions_weights.h5')


model_json = model_purity.to_json()
with open("model_purity.json", "w") as json_file:
 json_file.write(model_json)

model_purity.save_weights('purity_weights.h5')



# save and load model
json_file = open('wgd_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("wgd_weights.h5")
print("Loaded model from disk")



# save and load model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("wgd_weights.h5")
print("Loaded model from disk")



# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
true_p = 0
list_of_solutions = list()
for i, key in enumerate(test_keys):
    idx=np.argmin(np.abs(all_data.modes_dict[key]['alpha'] - test_purity_predictions[i]))
    if all_data.modes_dict[key]['solution_class'][idx] == 1:
        true_p = true_p + 1
    list_of_solutions.append(idx+1)
