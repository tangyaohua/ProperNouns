from __future__ import division
__author__ = 'tangyh'

#!/usr/bin/env python

import argparse
import cPickle
import gzip
import bz2
import logging
import os

import numpy
import tables

from collections import Counter
from operator import add
from numpy.lib.stride_tricks import as_strided
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="""
This takes a list of .txt or .txt.gz files and does word counting and
creating a dictionary (potentially limited by size). It uses this
dictionary to binarize the text into a numeric format (replacing OOV
words with 1) and create n-grams of a fixed size (padding the sentence
with 0 for EOS and BOS markers as necessary). The n-gram data can be
split up in a training and validation set.

The n-grams are saved to HDF5 format whereas the dictionary, word counts
and binarized text are all pickled Python objects.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-v", "--vocab", type=int, metavar="N",default=100,
                    help="limit vocabulary size to this number, which must "
                          "include BOS/EOS and OOV markers")
parser.add_argument("-t", "--char", action="store_true",
                    help="character-level processing")
parser.add_argument("-l", "--lowercase", action="store_true",
                    help="lowercase")
parser.add_argument('-a', '--align',default='preprocess/align',
                    help='the name of the align file')
parser.add_argument('-ts', '--target',default='preprocess/chinese',
                    help='the name of the target(chinese) file')
parser.add_argument('-ss', '--source',default='preprocess/english',
                    help='the name of the source(english) file')


def safe_pickle(obj, filename):
    if os.path.isfile(filename) and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def safe_hdf(array, name):
    if os.path.isfile(name + '.hdf') and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (name + '.hdf'))
    else:
        if os.path.isfile(name + '.hdf'):
            logger.info("Overwriting %s." % (name + '.hdf'))
        else:
            logger.info("Saving to %s." % (name + '.hdf'))
        with tables.openFile(name + '.hdf', 'w') as f:
            atom = tables.Atom.from_dtype(array.dtype)
            filters = tables.Filters(complib='blosc', complevel=5)
            ds = f.createCArray(f.root, name.replace('.', ''), atom,
                                array.shape, filters=filters)
            ds[:] = array


def create_dictionary(filename):
    # Part I: Counting the words
    counters = []
    sentence_counts = []
    global_counter = Counter()

    input_file = open(filename)

    input_filename = os.path.basename(input_file.name)
    logger.info("Counting words in %s" % input_filename)
    counter = Counter()
    sentence_count = 0
    for line in input_file:
        if args.lowercase:
            line = line.lower()
        words = None
        if args.char:
            words = list(line.strip().decode('utf-8'))
        else:
            words = line.strip().split(' ')
        counter.update(words)
        global_counter.update(words)


        sentence_count += 1
        logger.info("%d unique words in %d sentences with a total of %d words."
                    % (len(counter), sentence_count, sum(counter.values())))

    # Part II: Combining the counts
    combined_counter = global_counter

    # Part III: Creating the dictionary
    if args.vocab is not None:
        if args.vocab <= 2:
            logger.info('Building a dictionary with all unique words')
            args.vocab = len(combined_counter) + 2
        vocab_count = combined_counter.most_common(args.vocab - 2)
        logger.info("Creating dictionary of %s most common words, covering "
                    "%2.1f%% of the text."
                    % (args.vocab,
                       100.0 * sum([count for word, count in vocab_count]) /
                       sum(combined_counter.values())))
    else:
        logger.info("Creating dictionary of all words")
        vocab_count = counter.most_common()
    vocab = {'UNK': 1, '<s>': 0, '</s>': 0}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 2
    return vocab


def binarize():
    falign=open(args.align)

    cvocab=create_dictionary(args.target)
    evocab=create_dictionary(args.source)

    csents = open(args.target)
    esents = open(args.source)

    onetomany = Counter()
    srcunkcnt = Counter()
    srcunkpor = Counter()
    srcunkdis = Counter()
    srcunkall = Counter()
    hassrcalign = Counter()

    for csent,esent,algin in zip(csents, esents,falign):
        if args.lowercase:
            print 'lowercase'
            csent = csent.lower()
            esent = esent.lower()

        cwords = csent.strip().split(' ')
        ewords = esent.strip().split(' ')
        aligns = algin.strip().split(' ')
        alignsDict = {}
        for alignpair in aligns:
            try:
                c1,e1=map(int,alignpair.split(':'))
            except ValueError:
                print alignpair

            c1-=1
            e1-=1
            if c1 in alignsDict:
                alignsDict[c1].append(e1)
            else:
                alignsDict[c1]=[e1]

        binarized_sentence=[]

        for i in range(len(cwords)):
            ind=cvocab.get(cwords[i])
            if ind==None:
                if i in alignsDict:
                    hassrcalign.update(['%d'%1])

                    srcalign = alignsDict[i]
                    onetomany.update(["%d"%(len(srcalign))])

                    unknum=0
                    reldis=''
                    for j in srcalign:
                        reldis+=('_%d'%(j-i))
                        srcunkdis.update(["%d"%(j-i)])
                        if evocab.get(ewords[j])==None:
                            unknum+=1
                    srcunkcnt.update(['%d'%unknum])
                    srcunkpor.update(['%.3f'%(unknum/len(srcalign))])

                    srcunkall.update([reldis])
                else:
                    hassrcalign.update(['%d'%0])

    print 'if the target unk has align wrods in source, how many words does it align to: ', onetomany
    print 'the number of unk among the source words aligned to the target unk: ',srcunkcnt
    print 'the portion of unk among the source words aligned to the target unk: ',srcunkpor
    print 'the relative distances between a target unk and its align source words: ',srcunkdis
    print 'the relative distances pattern of a target unk and its align source words: ',srcunkall

    print 'whether the unk in target has align words in source: ',hassrcalign

onetomany = {'1': 205597, '2': 11648, '3': 316, '4': 18, '6': 1, '8': 1}
srcunkcnt = {'0': 110988, '1': 104143, '2': 2330, '3': 104, '4': 15, '6': 1}
srcunkpor = {'0.000': 110988, '1.000': 104587, '0.500': 1934, '0.333': 42, '0.667': 30}
srcunkdis = {'0': 15602, '1': 14902, '2': 13243, '3': 12596, '4': 11639, '5': 11054, '6': 10344, '7': 9545, '-1': 9093, '8': 8794, '9': 7796, '10': 7085, '-2': 7006, '11': 6236, '-3': 5945, '12': 5458, '13': 5137, '-4': 4697, '14': 4510, '15': 4043, '-5': 3976, '16': 3559, '-6': 3209, '17': 3177, '18': 2787, '-7': 2595, '19': 2401, '-8': 2168, '20': 2131, '21': 1913, '-9': 1829, '22': 1655, '-10': 1549, '23': 1433, '24': 1302, '-11': 1300, '25': 1184, '-12': 1091, '26': 976, '-13': 926, '27': 831, '-14': 740, '28': 712, '29': 676, '-15': 650, '30': 614, '-16': 602, '-17': 514, '31': 475, '32': 434, '34': 421, '-18': 413, '-19': 406, '33': 396, '-20': 341, '-21': 319, '35': 306, '36': 275, '-22': 272, '37': 259, '-24': 215, '38': 205, '-23': 201, '-26': 191, '39': 183, '-25': 177, '40': 163, '41': 160, '42': 156, '-27': 144, '-28': 136, '-29': 129, '-30': 123, '43': 122, '44': 120, '-31': 103, '46': 96, '45': 95, '-32': 94, '-33': 83, '47': 80, '-35': 70, '-34': 69, '48': 69, '-36': 68, '-37': 58, '50': 58, '49': 56, '51': 52, '-38': 48, '52': 46, '-44': 41, '-39': 40, '55': 40, '53': 37, '-45': 34, '-42': 31, '-41': 31, '54': 31, '57': 30, '56': 30, '-40': 28, '-46': 23, '59': 22, '-47': 22, '58': 21, '62': 21, '-43': 19, '60': 19, '-48': 18, '-49': 17, '-51': 17, '64': 16, '61': 15, '63': 15, '-50': 15, '66': 13, '-55': 11, '65': 10, '67': 10, '68': 10, '-52': 10, '-54': 9, '-53': 8, '69': 7, '-59': 7, '71': 6, '-57': 6, '-56': 6, '-60': 5, '74': 5, '-61': 4, '-63': 4, '75': 4, '72': 4, '81': 3, '82': 3, '73': 3, '70': 3, '78': 3, '-67': 3, '-66': 3, '-71': 2, '-88': 2, '93': 2, '-62': 2, '-65': 2, '-68': 2, '76': 2, '-72': 1, '-70': 1, '80': 1, '86': 1, '-86': 1, '77': 1, '79': 1, '-58': 1}
srcunkall = {'_0': 14131, '_1': 13191, '_2': 11727, '_3': 11150, '_4': 10278, '_5': 9776, '_6': 9092, '_7': 8430, '_-1': 8147, '_8': 7848, '_9': 6944, '_-2': 6311, '_10': 6279, '_11': 5520, '_-3': 5415, '_12': 4863, '_13': 4640, '_-4': 4271, '_14': 4042, '_-5': 3619, '_15': 3564, '_16': 3163, '_-6': 2914, '_17': 2818, '_18': 2466, '_-7': 2340, '_19': 2165, '_-8': 1964, '_20': 1929, '_21': 1708, '_-9': 1670, '_22': 1441, '_-10': 1403, '_23': 1270, '_-11': 1175, '_24': 1168, '_25': 1047, '_-12': 996, '_0_1': 889, '_26': 871, '_-13': 859, '_1_2': 762, '_27': 758, '_3_4': 703, '_2_3': 694, '_-14': 687, '_28': 655, '_5_6': 618, '_29': 616, '_4_5': 611, '_-15': 597, '_6_7': 583, '_-16': 558, '_30': 552, '_-1_0': 530, '_7_8': 486, '_-17': 476, '_31': 426, '_8_9': 420, '_32': 405, '_9_10': 393, '_-19': 387, '_-18': 387, '_34': 383, '_-2_-1': 379, '_10_11': 379, '_33': 363, '_-20': 315, '_11_12': 310, '_-21': 296, '_-3_-2': 284, '_35': 277, '_12_13': 268, '_-22': 258, '_36': 255, '_37': 239, '_14_15': 237, '_-4_-3': 223, '_15_16': 213, '_-24': 208, '_13_14': 205, '_-23': 193, '_-5_-4': 188, '_38': 187, '_-26': 179, '_17_18': 177, '_39': 171, '_-25': 168, '_16_17': 156, '_40': 155, '_-6_-5': 154, '_41': 151, '_42': 146, '_-27': 132, '_-7_-6': 125, '_18_19': 123, '_-28': 120, '_-29': 115, '_43': 114, '_-8_-7': 113, '_-30': 113, '_44': 111, '_21_22': 103, '_19_20': 98, '_22_23': 98, '_-31': 98, '_20_21': 92, '_46': 91, '_-32': 89, '_45': 87, '_-9_-8': 81, '_-33': 76, '_47': 74, '_24_25': 71, '_-10_-9': 70, '_-11_-10': 69, '_-35': 66, '_-36': 65, '_-34': 64, '_48': 63, '_25_26': 60, '_-37': 57, '_50': 54, '_23_24': 52, '_49': 52, '_-12_-11': 50, '_51': 50, '_-38': 47, '_52': 45, '_26_27': 41, '_-13_-12': 40, '_55': 40, '_-39': 39, '_-44': 36, '_53': 35, '_-45': 31, '_30_31': 31, '_29_30': 30, '_-41': 30, '_54': 30, '_-42': 29, '_56': 28, '_27_28': 28, '_28_29': 28, '_57': 27, '_-40': 27, '_-15_-14': 26, '_0_1_2': 26, '_-16_-15': 25, '_-14_-13': 24, '_33_34': 21, '_59': 21, '_-46': 21, '_-47': 21, '_62': 20, '_1_2_3': 20, '_58': 19, '_60': 19, '_-18_-17': 19, '_-17_-16': 19, '_-48': 18, '_-49': 17, '_6_7_8': 17, '_31_32': 17, '_34_35': 17, '_4_5_6': 16, '_9_10_11': 16, '_61': 15, '_-51': 15, '_3_4_5': 15, '_-21_-20': 14, '_5_6_7': 14, '_-43': 14, '_64': 13, '_-1_0_1': 13, '_2_3_4': 13, '_-50': 13, '_-2_-1_0': 12, '_32_33': 12, '_-20_-19': 12, '_37_38': 11, '_63': 11, '_66': 11, '_35_36': 11, '_13_14_15': 11, '_-3_-2_-1': 11, '_7_8_9': 11, '_8_9_10': 11, '_65': 10, '_-52': 10, '_68': 9, '_-22_-21': 9, '_-55': 9, '_16_17_18': 9, '_-4_-3_-2': 8, '_22_23_24': 8, '_36_37': 8, '_-53': 8, '_-7_-6_-5': 8, '_38_39': 7, '_67': 7, '_69': 7, '_-29_-28': 7, '_-54': 7, '_-59': 7, '_-19_-18': 7, '_71': 6, '_41_42': 6, '_10_11_12': 6, '_14_15_16': 6, '_-57': 6, '_-56': 6, '_39_40': 5, '_19_20_21': 5, '_-8_-7_-6': 5, '_-23_-22': 5, '_17_18_19': 5, '_-26_-25': 5, '_-28_-27': 5, '_-27_-26': 5, '_-60': 5, '_74': 5, '_44_45': 5, '_-30_-29': 5, '_15_16_17': 5, '_-63': 4, '_-9_-8_-7': 4, '_-25_-24': 4, '_43_44': 4, '_-61': 4, '_-5_-4_-3': 4, '_11_12_13': 4, '_75': 4, '_72': 4, '_47_48': 4, '_42_43': 4, '_12_13_14': 4, '_45_46': 3, '_40_41': 3, '_81': 3, '_82': 3, '_-6_-5_-4': 3, '_-44_-43': 3, '_-11_-10_-9': 3, '_-12_-11_-10': 3, '_-67': 3, '_17_18_19_20': 3, '_4_5_6_7': 3, '_-24_-23': 3, '_25_26_27': 3, '_78': 3, '_73': 3, '_-66': 3, '_70': 3, '_13_14_15_16': 3, '_63_64': 3, '_-33_-32': 3, '_-31_-30': 3, '_-34_-33': 3, '_-51_-50': 2, '_-88': 2, '_12_13_14_15': 2, '_93': 2, '_48_49': 2, '_46_47': 2, '_66_67': 2, '_16_17_18_19': 2, '_-28_-27_-26': 2, '_23_24_25': 2, '_20_21_22_23': 2, '_56_57': 2, '_20_21_22': 2, '_-55_-54': 2, '_-36_-35': 2, '_-65': 2, '_-62': 2, '_-32_-31': 2, '_49_50': 2, '_-68': 2, '_15_16_17_18': 2, '_-71': 2, '_-45_-44': 2, '_-43_-42': 2, '_50_51': 2, '_-30_-29_-28': 2, '_76': 2, '_-86': 1, '_80': 1, '_-14_-13_-12': 1, '_-35_-34_-33': 1, '_21_22_23': 1, '_52_53': 1, '_-2_-1_0_1_2_3': 1, '_-70': 1, '_-39_-38': 1, '_-41_-40': 1, '_29_30_31': 1, '_-37_-36': 1, '_62_63': 1, '_-15_-14_-13_-12': 1, '_79': 1, '_24_25_26': 1, '_-15_-14_-13': 1, '_-58': 1, '_-35_-34': 1, '_-10_-9_-8': 1, '_58_59': 1, '_77': 1, '_53_54': 1, '_-47_-46': 1, '_5_6_7_8_9_10_11_12': 1, '_35_36_37': 1, '_27_28_29': 1, '_-46_-45': 1, '_-72': 1, '_57_58': 1, '_86': 1, '_67_68': 1}
hassrcalign = {'1': 217581, '0': 44013}



def plotpie(datadict,xlabel, title, perc=1.0):
    # for label, value in datadict.items():
    b =DataFrame.from_dict(datadict, orient='index')
    b= b.sort(column=0, ascending=False)

    b['perc']= b[0]/b[0].sum()
    b['allperc'] = b['perc'].cumsum()

    b=b[b.allperc<perc]

    print b

    b.perc.plot(kind='bar')
    plt.xlabel(xlabel)
    plt.ylabel('Percent')
    plt.title(title)
    plt.show()

# plotpie(onetomany,'number of source words the unk aligned to','if the target unk has align wrods in source, how many words does it align to',1)
# plotpie(srcunkcnt,'number of unk among the source words','the number of unk among the source words aligned to the target unk',1.2)
# plotpie(srcunkpor,'portion of unk among the source words','the portion of unk among the source words aligned to the target unk',1.2)
# plotpie(srcunkdis,'relative distance','the relative distances between a target unk and its align source words', 0.90)
# plotpie(srcunkall,'relative distance pattern','the relative distances pattern of a target unk and its align source words',0.90)
# plotpie(hassrcalign,'1->has, 0->none','whether the unk in target has align words in source',1.2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    print type(args.target)
    print args.target
    binarize()
