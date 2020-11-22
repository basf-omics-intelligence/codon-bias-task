import numpy as np
from tqdm import tqdm

# DNA codon - amino acid table
__CODON_DICT__ = {'AAT': 'N', 'AAC': 'N',
                  'AAA': 'K', 'AAG': 'K',
                  'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                  'AGT': 'S', 'AGC': 'S',
                  'AGA': 'R', 'AGG': 'R',
                  'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
                  'ATG': 'M',
                  'CAT': 'H', 'CAC': 'H',
                  'CAA': 'Q', 'CAG': 'Q',
                  'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                  'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
                  'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                  'GAT': 'D', 'GAC': 'D',
                  'GAA': 'E', 'GAG': 'E',
                  'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                  'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
                  'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                  'TAT': 'Y', 'TAC': 'Y',
                  'TAA': '*', 'TAG': '*',
                  'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                  'TGT': 'C', 'TGC': 'C',
                  'TGA': '*',
                  'TGG': 'W',
                  'TTT': 'F', 'TTC': 'F',
                  'TTA': 'L', 'TTG': 'L'}

__DNA_MAPPING__ = dict(zip('ACGT', range(4)))
__DNA2_MAPPING__ = {'A': (1,   0,   0,   0),
                    'C': (0,   1,   0,   0),
                    'G': (0,   0,   1,   0),
                    'T': (0,   0,   0,   1),
                    'N': (1/4, 1/4, 1/4, 1/4),
                    'R': (1/2, 0,   1/2, 0),
                    'Y': (0,   1/2, 0,   1/2),
                    'S': (0,   1/2, 1/2, 0),
                    'W': (1/2, 0,   0,   1/2),
                    'K': (0,   0,   1/2, 1/2),
                    'M': (1/2, 1/2, 0,   0),
                    'B': (0,   1/3, 1/3, 1/3),
                    'D': (1/3, 0,   1/3, 1/3),
                    'H': (1/3, 1/3, 0,   1/3),
                    'V': (1/3, 1/3, 1/3, 0),
                    '.': (0,   0,   0,   0),
                    '-': (0,   0,   0,   0)}

# __CODONS__ = [''.join(i) for i in itertools.product(['A', 'C', 'G', 'T'], repeat=3)]
__CODON_MAPPING__ = dict(zip(__CODON_DICT__.keys(), range(64)))
__AMINO_ACID_MAPPING__ = dict(zip('ARNDCQEGHILKMFPSTWYV*', range(21)))


def translate(seq, trl_dict=None):
    """
    Translate a nucleotide sequence string into an amino acid sequence string.

    :param trl_dict: a dictionary, mapping each possible codon to an amino-acid
    :param seq: a nucleotide sequence, as string
    :return: an amino acid sequence, as string
    """
    if trl_dict is None:
        trl_dict = __CODON_DICT__
    trl_seq = ''.join([trl_dict[seq[i:(i+3)]] for i in range(0, len(seq), 3)])

    return trl_seq


def one_hot_encode(seq, mapping):
    """
    One-hot encode a sequence

    :param seq: a sequence as string
    :param mapping: a dictionary, mapping each possible character/codon to an integer
    :return: one-hot encoded sequence as a numpy array with shape (sequence length, length of dictionary)
    """
    l = list(set(map(len, mapping.keys())))
    assert len(l) == 1, "Length of keys in mapping should be uniform"
    l = l[0]

    seq_enc = [mapping[seq[x:(x+l)]] for x in range(len(seq)) if x % l == 0]

    return np.eye(len(mapping))[seq_enc]


def one_hot_encode2(seq, mapping):
    """
    One-hot encode a sequence which may contain ambiguous characters/codons

    :param seq: a sequence as string
    :param mapping: a dictionary, mapping each possible representation of a nucleotide/codon to a 1-hot vector
    :return: one-hot encoded sequence as a numpy array with shape (sequence length, length of dictionary)
    """
    l = list(set(map(len, mapping.keys())))
    assert len(l) == 1, "Length of keys in mapping should be uniform"
    l = l[0]

    seq_enc = np.array([mapping[seq[x:(x+1)]] for x in range(len(seq)) if x % l == 0])
    return seq_enc


def one_hot_decode(seq, mapping):
    """
    Decode a one-hot encoded sequence back to a sequence as a string

    :param seq: a one-hot encoded sequence, as numpy array with shape (sequence length, length of dictionary)
    :param mapping: a dictionary, mapping each possible character/codon to an integer
    :return: a sequence as string
    """
    mapping_rev = dict((value, key) for key, value in mapping.items())

    seq_dec = ''.join([mapping_rev[x] for x in np.where(seq == 1.)[1]])
    return seq_dec


def one_hot_translation_mat(mapping1=None, mapping2=None, trl_dict=None):
    """
    Create a lookup matrix which can translate a one-hot encoded sequence of codons into a one-hot encoded sequence of
    amino acids

    :param mapping1: a dictionary, mapping each possible codon to an integer
    :param mapping2: a dictionary, mapping each possible amino-acid to an integer
    :param trl_dict: a dictionary, mapping each possible codon to an amino-acid
    :return: a lookup matrix as numpy array
    """
    if mapping1 is None:
        mapping1 = __CODON_MAPPING__
    if mapping2 is None:
        mapping2 = __AMINO_ACID_MAPPING__
    if trl_dict is None:
        trl_dict = __CODON_DICT__

    mat = np.zeros((len(mapping1), len(mapping2)))
    indices = np.array([[mapping1[codon], mapping2[amino_acid]] for codon, amino_acid in trl_dict.items()])
    for ix in indices:
        mat[ix[0], ix[1]] = 1.
    return mat


__TRL_MAT__ = one_hot_translation_mat()


def translate_one_hot(seq, trl_mat=None):
    """
    Translate a one-hot encoded codon sequence into a one-hot encoded amino acid sequence.
    :param seq:
    :param trl_mat:
    :return:
    """
    if trl_mat is None:
        trl_mat = __TRL_MAT__

    return np.dot(seq, trl_mat)


def padding_one_hot(seq_list, max_len=None, padding='post'):
    assert padding in ['pre', 'post']

    if max_len is None:
        max_len = np.max([len(seq) for seq in seq_list])

    seqs = []
    for i, seq in enumerate(tqdm(seq_list)):
        seq = seq.astype('bool')
        seq_len, k = seq.shape
        pad_len = max_len - seq_len
        paddings = np.zeros((pad_len, k), dtype='bool')
        if padding == 'pre':
            seq = np.concatenate((paddings, seq), axis=0)
        elif padding == 'post':
            seq = np.concatenate((seq, paddings), axis=0)
        seqs.append(seq)

    seqs_array = np.array(seqs)
    return seqs_array


# example
if __name__ == "__main__":
    # given nucleotide sequence
    dna = "ATGGAAGTATTTAAAGCGCCACCTATTGGGATATAA"
    print(f"given dna sequence: {dna}")

    # if no special characters in dna sequence you can 1hot encode in this way:
    dna_1hot = one_hot_encode(dna, __DNA_MAPPING__)
    print(f"dna_1hot shape: ({dna_1hot.shape[0]}, {dna_1hot.shape[1]})")
    # if there are characters other then A, C, G, or T you can 1-hot-encode like this
    dna_1hot_2 = one_hot_encode2(dna, __DNA2_MAPPING__)
    print(f"dna_1hot_2 shape: ({dna_1hot.shape[0]}, {dna_1hot.shape[1]})")

    # translate into an amino-acid sequence
    protein = translate(dna)
    print(f"dna sequence translated into protein: {protein}")

    # create one-hot encoded codon sequence
    codons_1hot = one_hot_encode(dna, __CODON_MAPPING__)
    print(f"codons_1hot shape: ({codons_1hot.shape[0]}, {codons_1hot.shape[1]})")

    # translate one-hot encoded codon sequence into one-hot encoded amino-acid sequence
    protein_1hot = translate_one_hot(codons_1hot)
    print(f"protein_1hot shape: ({protein_1hot.shape[0]}, {protein_1hot.shape[1]})")

    # decode one-hot encoded amino-acid sequence into an amino-acid sequence string
    protein_back = one_hot_decode(protein_1hot, __AMINO_ACID_MAPPING__)
    print(f"1-hot-encoded protein decoded into protein: {protein_back}")
