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
            c1,e1=map(int,alignpair.split(':'))
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

    print onetomany
    print srcunkcnt
    print srcunkpor
    print srcunkdis
    print srcunkall
    print hassrcalign


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    print type(args.target)
    print args.target
    binarize()
