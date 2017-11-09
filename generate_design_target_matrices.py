import pandas as pd
from scipy.stats import beta
import os
import argparse
import datetime
import numpy as np
import generate_basic_features as process_mut_acn


class feature_set_basic_regression:
    """input tsv file and corresponding features / targets"""

    def __init__(self, args):
        self.data_table = pd.read_csv(args.data_tsv, sep='\t')
        self.data_table.dropna(inplace=True)
        self.data_table.reset_index(inplace=True,drop=True)
        self.id = args.feature_set_id
        self.X = np.zeros([len(self.data_table), 3, 200])
        self.target = np.zeros([len(self.data_table),2])

    def validate_data_table(self):
        print 'Validating data table'
        remove_rows = np.ones([len(self.data_table),1],dtype=bool)
        for index,row in self.data_table.iterrows():
            if not os.path.isfile(row['absolute_annotated_maf']) or not os.path.isfile(row['absolute_seg_file']):
                remove_rows[index] = False
        print 'Removing ' + str(np.sum(~remove_rows)) + ' rows due to missing files'
        self.data_table = self.data_table[remove_rows]
        self.data_table.reset_index(inplace=True,drop=True)

    def generate_acna_matrix(self):
        nbin = 200
        print 'Generating basic copy number feature matrix on ' + str(len(self.data_table)) + ' samples'
        X_acna = np.zeros([len(self.data_table),2, nbin])
        for index, row in self.data_table.iterrows():
            if np.mod(self.data_table,100) == 0:
                print str(index)+'/'+str(len(self.data_table))
            X_acna[index,:, :] = process_mut_acn.read_and_process_seg_file(row['absolute_seg_file'], nbin)
        return X_acna

    def generate_mutation_matrix(self):
        X_mut = np.zeros([len(self.data_table), 100])
        print 'Generating basic mutation feature matrix on ' + str(len(self.data_table)) + ' samples'

        for index, row in self.data_table.iterrows():
            if np.mod(self.data_table,100) == 0:
                print str(index)+'/'+str(len(self.data_table))
            X_mut[index, :] = process_mut_acn.read_and_process_maf_file(row['absolute_annotated_maf'])
        return X_mut

    def generate_X(self):
        X_acna = self.generate_acna_matrix()
        X_mut = self.generate_mutation_matrix()
        self.X[:,0:2,:] = X_acna
        self.X[:,2,0:100] = X_mut

    def generate_target(self):
        self.target[:,0] = self.data_table['purity']
        self.target[:,1] = self.data_table['ploidy']


def main():
    """ read in a table of seg files and produce aCNA feature matrix for infering absolute (purity,ploidy) mode"""
    parser = argparse.ArgumentParser(description='Build design matrix for training ABSOLUTE solution picker')
    parser.add_argument('--data_tsv', help='TSV file with the following headers: pair_id,purity_absolute_reviewed,'
                                           'purity_absolute_reviewed,absolute_seg_file,'
                                           'absolute_annotated_maf',
                        required=True)
    parser.add_argument('--feature_set_id', help='file stem for feature set pickle files', required=True)

    args = parser.parse_args()

    basic_features = feature_set_basic_regression(args)
    basic_features.validate_data_table()
    basic_features.generate_X()
    basic_features.generate_target()

    np.savez(basic_features.id + '_' + str(datetime.datetime.now().time()), X=basic_features.X, Target=basic_features.target)

if __name__ == "__main__":
    main()