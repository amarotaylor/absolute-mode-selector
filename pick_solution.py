from argparse import Namespace
from sklearn.preprocessing import StandardScaler
import argparse
import generate_design_target_matrices as genmat
from keras.models import model_from_json
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def load_and_configure_keras_model(model_json,model_weights):
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights)
    return loaded_model

def read_and_generate_features(args):
    data = genmat.solution_prediction_features(args)
    data.validate_data_table()
    data.generate_X()
    return data

def createNumpyDataSet(data, keys, mode_num):
    np_data = Namespace()
    sample_num = len(keys)
    np_data.mut_mat = np.zeros([sample_num, 301])
    np_data.abs_mat = np.zeros([sample_num, len(data.pp_modes_tables[list(keys)[0]].columns) , mode_num])
    for i, key in enumerate(keys):
        np_data.mut_mat[i, :] = data.cna_mutation_array[key]
        modes_shape = data.pp_modes_tables[key].shape
        if modes_shape[0] > mode_num:  # truncate the modes according to predefined value
            np_data.abs_mat[i, :, :] = np.nan_to_num(np.transpose(data.pp_modes_tables[key].loc[:mode_num - 1, :].values))
        else:
            np_data.abs_mat[i, :, :modes_shape[0]] = np.nan_to_num(np.transpose(data.pp_modes_tables[key].loc[:, :].values))
    return np_data


def createModeClassificationDataSet(data,purity_pred,wgd_pred, keys):
    X_modes = []
    x_new = data.pp_modes_tables[keys[0]]
    pred_purity = np.zeros(len(x_new)) + purity_pred[0]
    pred_wgd = np.zeros(len(x_new)) + wgd_pred[0]
    x_new['pred_purity'] = pred_purity
    x_new['pred_wgd'] = pred_wgd
    x_new['purity_diff'] = np.abs(x_new['pred_purity'] - x_new['alpha'])
    X_modes = x_new
    X_modes.dropna(axis=1, how='all', inplace=True)
    X_array = X_modes.reset_index().values
    return X_array

def main():
    """ read in a table of seg files and produce aCNA feature matrix for infering absolute (purity,ploidy) mode"""
    parser = argparse.ArgumentParser(description='Pick solution from ABSOLUTE inputs:')
    parser.add_argument('--r_data_file', help='Rdata file generated by solution containing solution data',required=True)
    parser.add_argument('--maf_file', help='maf file with t_alt_count and t_ref_count columns', required=True)
    parser.add_argument('--seg_file',help='ACNV seg file',required=True)
    parser.add_argument('--pair_id', help = 'sample_id',required=True)
    args = parser.parse_args()
    data_tsv = {}
    data_tsv['absolute_seg_file'] = args.seg_file
    data_tsv['absolute_annotated_maf'] = args.maf_file
    data_tsv['absolute_summary_data'] = args.r_data_file
    data_tsv['pair_id'] = args.pair_id
    data_tsv = pd.DataFrame(data_tsv, index=[0])
    data_tsv.to_csv('tmp.input', sep='\t', index=False)
    args_input = Namespace()
    args_input.data_tsv = 'tmp.input'
    args_input.feature_set_id = args.pair_id
    data = read_and_generate_features(args_input)
    wgs_model = load_and_configure_keras_model('model_wgd.json','wgd_weights.h5')
    purity_model = load_and_configure_keras_model('model_purity.json','purity_weights.h5')
    solution_model = load_and_configure_keras_model('model_solutions.json','solutions_weights.h5')
    key = list(data.data_mat[0].keys())
    np_data = createNumpyDataSet(data, key, 20)
    predicted_purity = purity_model.predict(np_data.mut_mat)
    wgd_scaler = joblib.load('wgd_scaler.pkl')
    mut_mat_scale = wgd_scaler.transform(np_data.mut_mat)
    predicted_wgd = wgs_model.predict(mut_mat_scale)
    solutions_scaler = joblib.load('solutions_scaler.pkl')
    solution_data = createModeClassificationDataSet(data, predicted_purity, predicted_wgd, key)
    solution_data_scale = solutions_scaler.transform(solution_data)
    solutions = solution_model.predict(solution_data_scale) > 0.1
    if np.sum(solutions) > 0:
        true_val = np.argmax(solution_model.predict(solution_data_scale) > 0.05)
    else :
        true_val = np.nan

    with open("Output.txt", "w") as text_file:
        text_file.write(np.str(true_val))
        text_file.write('\n')



if __name__ == "__main__":
    main()