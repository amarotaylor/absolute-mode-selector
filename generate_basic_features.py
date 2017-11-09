import numpy as np
import sys
import pandas as pd
from scipy.stats import beta

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_member(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, np.nan) for itm in a]


def hg19_to_linear_positions(chromosome, position, **keyword_parameters):
    # type: (nparray, nparray,string) -> nparray
    """
    Change chromosome-position to continuous linear coordinates
    """
    if ('build' in keyword_parameters):
        build = keyword_parameters['build']
    else:
        build = 'hg19'
    if build == 'hg19':
        L = np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663,
                      146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540,
                      102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566,
                      155270560, 59373566, 16569])  # chromosome lengths from genome-mysql.cse.ucsc.edu
    if build == 'hg38':
        # add support for hg38
        sys.exit('support still missing for hg38')

    C = np.append(1, np.cumsum(L))
    x = np.array([chromosome[int(i)] for i in np.arange(0, len(position))])
    return C[[x.astype(int)]] + position

def chr2num(chr):
    # convert chromosome from strings to ints
    chr[chr == 'X'] = '23'
    chr[chr == 'Y'] = '24'
    chr[np.array(chr == 'MT') | np.array(chr == 'M')] = '25'
    chromosomes = np.array(range(1, 26))
    return np.array(is_member(chr, chromosomes.astype(np.str)))


def remove_small_centromere_segments(sample_seg):
    centromere_positions = [125000001, 1718373143, 1720573143, 1867507890,
                            1869607890, 1984214406, 1986714406, 2101066301,
                            2102666301, 2216036179, 2217536179, 2323085719, 2326285719, 2444417111,
                            2446417111, 2522371864, 2524171864, 2596767074, 2598567074, 2683844322,
                            2685944322, 339750621, 342550621, 2744173305, 2746073305, 2792498825, 2794798825,
                            2841928720,
                            2844428720, 580349994, 583449994, 738672424, 740872424, 927726700, 930026700, 1121241960,
                            1123541960, 1291657027, 1293557027, 1435895690, 1438395690, 1586459712, 1588159712]

    distance_centromere = np.zeros([len(sample_seg['genomic_coord_end']), len(centromere_positions)])
    for i, centromere in enumerate(centromere_positions):
        distance_centromere[:, i] = np.abs(sample_seg['genomic_coord_start'] - centromere) + np.abs(
            sample_seg['genomic_coord_end'] - centromere)
    distance_centromere = np.min(distance_centromere, axis=1)
    sample_seg = sample_seg[distance_centromere > 50000]
    sample_seg.dropna(inplace=True)
    sample_seg.reset_index(inplace=True, drop=True)
    return sample_seg


def read_and_process_seg_file(path,nbin):
    """ Read in segment file and bin the allelic copy number data into nbin number of sections.
    we use 20 as the max value bin set to contain the data from an inital 7000 sample cohort.
    Values of hscr exceeding 20 are likely artifacts. """
    sample_seg = pd.read_csv(path, sep='\t')
    sample_seg.dropna(inplace=True)
    sample_seg.reset_index(inplace=True,drop=True)



    if not is_number(sample_seg['Chromosome'][0]):
            sample_seg['contig'] = chr2num(sample_seg['Chromosome'])
    else :
        sample_seg['contig'] = sample_seg['Chromosome']

    sample_seg['genomic_coord_start'] = hg19_to_linear_positions(sample_seg['contig'], sample_seg['Start.bp'])
    sample_seg['genomic_coord_end'] = hg19_to_linear_positions(sample_seg['contig'], sample_seg['End.bp'])

    sample_seg = remove_small_centromere_segments(sample_seg)
    # see comment above
    sample_seg.loc[sample_seg['hscr.a1']> int(np.true_divide(nbin,10)), 'hscr.a1'] = int(np.true_divide(nbin,10))
    sample_seg.loc[sample_seg['hscr.a2']> int(np.true_divide(nbin,10)), 'hscr.a2'] = int(np.true_divide(nbin,10))

    X_acna = np.zeros([2,nbin])
    W = np.zeros([sample_seg.shape[0], 1])

    total_territory = 0
    for index, row in sample_seg.iterrows():
        total_territory += sample_seg['End.bp'][index] - sample_seg['Start.bp'][index]

    for index, row in sample_seg.iterrows():
        seg_length = float(sample_seg['End.bp'][index] - sample_seg['Start.bp'][index])
        W[index] = seg_length / total_territory
        X_acna[0,int(np.round(sample_seg['hscr.a1'][index], 1) * 10)] += W[index]
        X_acna[1,int(np.round(sample_seg['hscr.a2'][index], 1) * 10)] += W[index]
    return X_acna

def read_and_process_maf_file(path):
    """ Read in maf file and generate 100x1 mutation based features.
    Feature is just summed beta pdfs from all mutations observed in the sample.
    The presence of higher allele fraction events is consistent with greater purity."""
    fields = ['alt', 'ref']
    sample_maf = pd.read_csv(path, sep="\t", usecols=fields)

    x = np.linspace(beta.ppf(0, 1, 1), beta.ppf(1, 1, 1), 100)
    X_mut = np.zeros([1,len(x)])

    for index, row in sample_maf.iterrows():
        X_mut += beta.pdf(x, row['alt'] + 1, row['ref'] + 1)
    return X_mut
