import requests
import urllib.request
from io import BytesIO
import gzip
import pandas as pd
import re
import os
from Bio import SeqIO

EXPRESSION_URL = "https://www.ebi.ac.uk/gxa/experiments-content/E-CURD-1/resources/" \
                 "ExperimentDownloadSupplier.RnaSeqBaseline/tpms.tsv"
CDS_URL = "https://www.arabidopsis.org/download_files/Sequences/Araport11_blastsets/Araport11_genes.201606.cds.fasta.gz"


def download_expression_data(fn=None):
    url = EXPRESSION_URL
    r = requests.get(url)
    if fn is None:
        fn = "data/raw/" + re.findall("filename=\"(.+)\"", r.headers['content-disposition'])[0]

    print("Downloading from " + url)
    with open(fn, 'wb') as f:
        f.write(r.content)
    print(fn)
    return fn


def download_cds_data(fn=None):
    url = CDS_URL

    print("Downloading from " + url)
    response = urllib.request.urlopen(CDS_URL)
    compressed_file = BytesIO(response.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)

    if fn is None:
        fn = "data/raw/" + os.path.basename(url).replace('.gz', '')

    with open(fn, 'wb') as f:
        f.write(decompressed_file.read())
    print(fn)
    return fn


def load_expression_data(fn):
    expr_dat = pd.read_csv(fn, sep='\t', skiprows=4)
    return expr_dat


def load_cds_data(fn):
    res = {'Gene ID': list(),
           'Transcript ID': list(),
           'Length': list(),
           'CDS sequence': list()}

    with open(fn) as f:
        for seq_record in SeqIO.parse(f, 'fasta'):
            res['Gene ID'].append(seq_record.id.split('.')[0])
            res['Transcript ID'].append(seq_record.id)
            res['Length'].append(len(seq_record.seq))
            res['CDS sequence'].append(str(seq_record.seq))

    return pd.DataFrame.from_dict(res)


def process_data(expr_df, cds_df, save=True, save_fn='data/processed/cds_expr.txt'):
    # calculate maximal expression accross samples
    cols = list(expr_df.columns[2:])
    expr_df['Max TPM'] = expr_df[cols].max(axis=1)
    expr_df = expr_df.drop(cols, axis=1)

    # only keep cds with length divisible by 3
    select_ix = cds_df['Length'] % 3 == 0
    cds_df = cds_df[select_ix]

    # only keep cds which only contain A, C, G, or T
    select_ix = cds_df['CDS sequence'].transform(set) <= {'C', 'A', 'T', 'G'}
    cds_df = cds_df[select_ix]

    # only keep the longest transcripts cds per gene
    select_ix = cds_df.groupby(['Gene ID'])['Length'].transform(max) == cds_df['Length']
    cds_df = cds_df[select_ix]
    # if multiple transcript cds with the same length, take first
    cds_df = cds_df.groupby('Gene ID').first().reset_index()

    # merge expression and cds sequences
    merged = pd.merge(expr_df, cds_df, how="inner", on="Gene ID")

    if save and save_fn is not None:
        merged.to_csv(save_fn, index=False, sep='\t')
        print(save_fn)
    return merged


if __name__ == "__main__":
    expr_fn = download_expression_data()
    expr_data = load_expression_data(expr_fn)
    cds_fn = download_cds_data()
    cds_data = load_cds_data(cds_fn)
    cds_expr = process_data(expr_data, cds_data)
    pass
