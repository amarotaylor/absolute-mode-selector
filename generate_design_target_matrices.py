import pandas as pd
import os
import argparse
import datetime
import numpy as np
import generate_basic_features as process_mut_acn
import cPickle as pkl
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage


class feature_set_basic_regression:
    """input tsv file and corresponding features / targets"""

    def __init__(self, args):
        self.data_table = pd.read_csv(args.data_tsv, sep='\t')
        self.data_table.dropna(inplace=True)
        self.data_table.reset_index(inplace=True, drop=True)
        self.id = args.feature_set_id

    def validate_data_table(self):
        print 'Validating data table'
        remove_rows = np.ones([len(self.data_table), 1], dtype=bool)
        for index, row in self.data_table.iterrows():
            if not os.path.isfile(row['absolute_annotated_maf']) or not os.path.isfile(row['absolute_seg_file']):
                remove_rows[index] = False
        print 'Removing ' + str(np.sum(~remove_rows)) + ' rows due to missing files'
        self.data_table = self.data_table[remove_rows]
        self.data_table.reset_index(inplace=True, drop=True)
        self.X = np.zeros([len(self.data_table), 301])
        self.target = np.zeros([len(self.data_table), 2])

    def generate_acna_matrix(self):
        nbin = 200
        print 'Generating basic copy number feature matrix on ' + str(len(self.data_table)) + ' samples'
        X_acna = np.zeros([len(self.data_table), nbin])
        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            X_acna[index, :] = process_mut_acn.read_and_process_seg_file(row['absolute_seg_file'], nbin)
        return X_acna

    def generate_mutation_matrix(self):
        X_mut = np.zeros([len(self.data_table), 100])
        print 'Generating basic mutation feature matrix on ' + str(len(self.data_table)) + ' samples'

        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            X_mut[index, :] = process_mut_acn.read_and_process_maf_file(row['absolute_annotated_maf'])
        return X_mut

    def generate_X(self):
        X_acna = self.generate_acna_matrix()
        X_mut = self.generate_mutation_matrix()
        self.X[:, 0:200] = X_acna
        self.X[:, 201:301] = X_mut

    def generate_target(self):
        self.target[:, 0] = self.data_table['purity']
        self.target[:, 1] = self.data_table['ploidy']


class solution_classification_features:
    """input tsv file and corresponding features / targets"""

    def __init__(self, args):
        self.data_table = pd.read_csv(args.data_tsv, sep='\t')
        self.data_table.dropna(inplace=True)
        self.data_table.reset_index(inplace=True, drop=True)
        self.id = args.feature_set_id
        self.pp_modes_tables = dict()
        self.cna_mutation_array = dict()
        self.target_purity = dict()
        self.target_ploidy = dict()

    def validate_data_table(self):
        print 'Validating data table'
        remove_rows = np.ones([len(self.data_table), 1], dtype=bool)
        for index, row in self.data_table.iterrows():
            if not os.path.isfile(row['absolute_annotated_maf']) or not os.path.isfile(row['absolute_seg_file']):
                remove_rows[index] = False
        print 'Removing ' + str(np.sum(~remove_rows)) + ' rows due to missing files'
        self.data_table = self.data_table[remove_rows]
        self.data_table.reset_index(inplace=True, drop=True)
        self.X = np.zeros([len(self.data_table), 301])
        self.target = np.zeros([len(self.data_table), 2])

    def generate_solutions_tables(self):
        ''' code from Adam use rpy2 to execute rcode which reads out a solutions file to pandas '''
        col_names = ['alpha', 'tau', 'AT', 'b', 'delta', 'LL', 'mode_curv', 'genome mass', 'sigma.h.hat', 'theta.z.hat',
                     'sigma.A.hat',
                     'theta.Q.hat', 'lambda.hat', 'theta.0', 'frac.het', 'SCNA_LL', 'entropy', 'Kar_LL', 'WGD',
                     'combined_LL',
                     'SSNV_LL', 'SCNA_Theta_integral', 'dens']

        # Build R function to be used as a python package
        load_RData_func_str = """
                   load_RData <- function(file_path) {
                      load(file_path)
                      head_name <- ls()[1]
                      file_name <- names(`segobj.list`)[1]
                      r_data <- `segobj.list`[[file_name]]$mode.res$mode.tab
                      return(r_data)
                  }
                  """
        # Pack the function above as a package
        r_pack = SignatureTranslatedAnonymousPackage(load_RData_func_str, "r_pack")
        print 'Generating absolute tables for ' + str(len(self.data_table)) + ' samples'
        pandas2ri.activate()
        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            r_data = r_pack.load_RData(row['absolute_summary_data'])
            abs_table = pd.DataFrame(pandas2ri.ri2py(r_data), columns=col_names)
            abs_table_solution_class = np.zeros([len(abs_table)])
            abs_table_solution_class[np.argmin(np.abs(abs_table['alpha'] - row['purity']) + np.abs(
                abs_table['tau'] - row['ploidy']))] = 1
            abs_table['solution_class'] = abs_table_solution_class
            self.pp_modes_tables[row['pair_id']] = abs_table

        pandas2ri.deactivate()

    def generate_cna_mut_matrix(self):
        nbin = 200
        print 'Generating copy number and mutation feature matrix on ' + str(len(self.data_table)) + ' samples'
        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            X_array = np.zeros([301])
            X_array[0:200] = process_mut_acn.read_and_process_seg_file(row['absolute_seg_file'], nbin)
            X_array[201:301] = process_mut_acn.read_and_process_maf_file(row['absolute_annotated_maf'])
            self.cna_mutation_array[row['pair_id']] = X_array

    def generate_X(self):
        self.generate_cna_mut_matrix()
        self.generate_solutions_tables()
        self.generate_target()
        self.data_mat = [self.cna_mutation_array, self.pp_modes_tables, self.target_purity, self.target_ploidy]

    def generate_target(self):
        for index, row in self.data_table.iterrows():
            self.target_purity[row['pair_id']] = self.data_table['purity'][index]
            self.target_ploidy[row['pair_id']] = self.data_table['ploidy'][index]


class solution_prediction_features:
    """input tsv file and corresponding features """

    def __init__(self, args):
        self.data_table = pd.read_csv(args.data_tsv, sep='\t')
        self.data_table.dropna(inplace=True)
        self.data_table.reset_index(inplace=True, drop=True)
        self.id = args.feature_set_id
        self.pp_modes_tables = dict()
        self.cna_mutation_array = dict()
        self.target_purity = dict()
        self.target_ploidy = dict()

    def validate_data_table(self):
        print 'Validating data table'
        remove_rows = np.ones([len(self.data_table), 1], dtype=bool)
        for index, row in self.data_table.iterrows():
            if not os.path.isfile(row['absolute_annotated_maf']) or not os.path.isfile(row['absolute_seg_file']):
                remove_rows[index] = False
        print 'Removing ' + str(np.sum(~remove_rows)) + ' rows due to missing files'
        self.data_table = self.data_table[remove_rows]
        self.data_table.reset_index(inplace=True, drop=True)
        self.X = np.zeros([len(self.data_table), 301])
        self.target = np.zeros([len(self.data_table), 2])

    def generate_solutions_tables(self):
        ''' code from Adam use rpy2 to execute rcode which reads out a solutions file to pandas '''
        col_names = ['alpha', 'tau', 'AT', 'b', 'delta', 'LL', 'mode_curv', 'genome mass', 'sigma.h.hat',
                     'theta.z.hat',
                     'sigma.A.hat',
                     'theta.Q.hat', 'lambda.hat', 'theta.0', 'frac.het', 'SCNA_LL', 'entropy', 'Kar_LL', 'WGD',
                     'combined_LL',
                     'SSNV_LL', 'SCNA_Theta_integral', 'dens']

        # Build R function to be used as a python package
        load_RData_func_str = """
                       load_RData <- function(file_path) {
                          load(file_path)
                          head_name <- ls()[1]
                          file_name <- names(`segobj.list`)[1]
                          r_data <- `segobj.list`[[file_name]]$mode.res$mode.tab
                          return(r_data)
                      }
                      """
        # Pack the function above as a package
        r_pack = SignatureTranslatedAnonymousPackage(load_RData_func_str, "r_pack")
        print 'Generating absolute tables for ' + str(len(self.data_table)) + ' samples'
        pandas2ri.activate()
        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            r_data = r_pack.load_RData(row['absolute_summary_data'])
            abs_table = pd.DataFrame(pandas2ri.ri2py(r_data), columns=col_names)
            self.pp_modes_tables[row['pair_id']] = abs_table
        pandas2ri.deactivate()

    def generate_cna_mut_matrix(self):
        nbin = 200
        print 'Generating copy number and mutation feature matrix on ' + str(len(self.data_table)) + ' samples'
        for index, row in self.data_table.iterrows():
            if np.mod(index, 100) == 0:
                print str(index) + '/' + str(len(self.data_table))
            X_array = np.zeros([301])
            X_array[0:200] = process_mut_acn.read_and_process_seg_file(row['absolute_seg_file'], nbin)
            X_array[201:301] = process_mut_acn.read_and_process_maf_file(row['absolute_annotated_maf'])
            self.cna_mutation_array[row['pair_id']] = X_array

    def generate_X(self):
        self.generate_cna_mut_matrix()
        self.generate_solutions_tables()
        self.data_mat = [self.cna_mutation_array, self.pp_modes_tables]



def main():
    """ read in a table of seg files and produce aCNA feature matrix for infering absolute (purity,ploidy) mode"""
    parser = argparse.ArgumentParser(description='Build design matrix for training ABSOLUTE solution picker')
    parser.add_argument('--data_tsv', help='TSV file with the following headers: pair_id,purity_absolute_reviewed,'
                                           'purity_absolute_reviewed,absolute_seg_file,'
                                           'absolute_annotated_maf',
                        required=True)
    parser.add_argument('--feature_set_id', help='file stem for feature set pickle files', required=True)
    parser.add_argument('--feature_type', help='basic or basic+ (with absolute tables)', required=True)
    args = parser.parse_args()
    if args.feature_type == 'basic':
        data = feature_set_basic_regression(args)
        data.validate_data_table()
        data.generate_X()
        data.generate_target()
    if args.feature_type == 'basic+':
        data = solution_classification_features(args)
        data.validate_data_table()
        data.generate_X()
        data.generate_target()

    pkl.dump(data.data_mat,
             open(data.id + '_' + args.feature_type + '_' + str(datetime.datetime.now().time()) + '.pickle', 'wb'))


if __name__ == "__main__":
    main()
