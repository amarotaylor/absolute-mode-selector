import pandas as pd
from scipy.stats import beta
import os
import argparse
import datetime

def main():
    """ read in a table of seg files and produce aCNA feature matrix for infering absolute (purity,ploidy) mode"""
    parser = argparse.ArgumentParser(description='Build design matrix for training ABSOLUTE solution picker')
    parser.add_argument('--sample_tsv', help='TSV file with the following headers: pair_id,purity_absolute_reviewed,'
                                                'purity_absolute_reviewed,absolute_seg_file,'
                                             'absolute_annotated_maf/absolute_annotated_maf_wgs',
                                            required = True)
    parser.add_argument('--feature_set_id',help='file stem for feature set pickle files',require=True)

