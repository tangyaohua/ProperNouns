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
parser.add_argument("input", type=argparse.FileType('r'), nargs="+",
                    help="The input files")
parser.add_argument("-b", "--binarized-text", default='binarized_text.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-d", "--dictionary", default='vocab.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-n", "--ngram", type=int, metavar="N",
                    help="create n-grams")
parser.add_argument("-v", "--vocab", type=int, metavar="N",
                    help="limit vocabulary size to this number, which must "
                          "include BOS/EOS and OOV markers")
parser.add_argument("-p", "--pickle", action="store_true",
                    help="pickle the text as a list of lists of ints")
parser.add_argument("-s", "--split", type=float, metavar="N",
                    help="create a validation set. If >= 1 take this many "
                         "samples for the validation set, if < 1, take this "
                         "fraction of the samples")
parser.add_argument("-o", "--overwrite", action="store_true",
                    help="overwrite earlier created files, also forces the "
                         "program not to reuse count files")
parser.add_argument("-e", "--each", action="store_true",
                    help="output files for each separate input file")
parser.add_argument("-c", "--count", action="store_true",
                    help="save the word counts")
parser.add_argument("-t", "--char", action="store_true",
                    help="character-level processing")
parser.add_argument("-l", "--lowercase", action="store_true",
                    help="lowercase")
parser.add_argument('-a', '--align',default='align',
                    help='the name of the align file')
parser.add_argument('-z','--target',action="store_true",
                    help='whether the text is target')


def open_files():
    base_filenames = []
    for i, input_file in enumerate(args.input):
        dirname, filename = os.path.split(input_file.name)
        if filename.split(os.extsep)[-1] == 'gz':
            base_filename = filename.rstrip('.gz')
        elif filename.split(os.extsep)[-1] == 'bz2':
            base_filename = filename.rstrip('.bz2')
        else:
            base_filename = filename
        if base_filename.split(os.extsep)[-1] == 'txt':
            base_filename = base_filename.rstrip('.txt')
        if filename.split(os.extsep)[-1] == 'gz':
            args.input[i] = gzip.GzipFile(input_file.name, input_file.mode,
                                          9, input_file)
        elif filename.split(os.extsep)[-1] == 'bz2':
            args.input[i] = bz2.BZ2File(input_file.name, input_file.mode)
        base_filenames.append(base_filename)
    return base_filenames


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


def create_dictionary():
    # Part I: Counting the words
    counters = []
    sentence_counts = []
    global_counter = Counter()

    for input_file, base_filename in zip(args.input, base_filenames):
        count_filename = base_filename + '.count.pkl'
        input_filename = os.path.basename(input_file.name)
        if os.path.isfile(count_filename) and not args.overwrite:
            logger.info("Loading word counts for %s from %s"
                        % (input_filename, count_filename))
            with open(count_filename, 'rb') as f:
                counter = cPickle.load(f)
            sentence_count = sum([1 for line in input_file])
        else:
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
        counters.append(counter)
        sentence_counts.append(sentence_count)
        logger.info("%d unique words in %d sentences with a total of %d words."
                    % (len(counter), sentence_count, sum(counter.values())))
        if args.each and args.count:
            safe_pickle(counter, count_filename)
        input_file.seek(0)

    # Part II: Combining the counts
    combined_counter = global_counter
    logger.info("Total: %d unique words in %d sentences with a total "
                "of %d words."
                % (len(combined_counter), sum(sentence_counts),
                   sum(combined_counter.values())))
    if args.count:
        safe_pickle(combined_counter, 'combined.count.pkl')

    # Part III: Creating the dictionary
    if args.vocab is not None:
        if args.vocab <= 2:
            logger.info('Building a dictionary with all unique words')
            args.vocab = len(combined_counter) + 2
        if args.target:
            vocab_count = combined_counter.most_common(args.vocab - 17)
        else:
            vocab_count = combined_counter.most_common(args.vocab - 2)
        logger.info("Creating dictionary of %s most common words, covering "
                    "%2.1f%% of the text."
                    % (args.vocab,
                       100.0 * sum([count for word, count in vocab_count]) /
                       sum(combined_counter.values())))
    else:
        logger.info("Creating dictionary of all words")
        vocab_count = counter.most_common()
    if args.target:
        vocab = {'unkpos1': 1, 'unkpos_1': 2, 'unkpos2': 3, 'unkpos_2': 4, 'unkpos3': 5, 'unkpos_3': 6,
                 'unkpos4': 7, 'unkpos_4': 8, 'unkpos5': 9, 'unkpos_5': 10, 'unkpos6': 11, 'unkpos_6': 12,
                 'unkpos7': 13, 'unkpos_7': 14, 'unkposn': 15, 'unkpos0': 16, '<s>': 0, '</s>': 0}
        for i, (word, count) in enumerate(vocab_count):
            vocab[word] = i + 17
    else:
        vocab = {'unk': 1, '<s>': 0, '</s>': 0}
        for i, (word, count) in enumerate(vocab_count):
            vocab[word] = i + 2
    # for (k,v) in vocab.items():
    #     print k,':',v
    safe_pickle(vocab, args.dictionary)
    return combined_counter, sentence_counts, counters, vocab


def binarize():
    # if target and args.align:
    falign=open(args.align)
    binarized_corpora = []
    total_ngram_count = 0
    for input_file, base_filename, sentence_count in \
            zip(args.input, base_filenames, sentence_counts):
        input_filename = os.path.basename(input_file.name)
        logger.info("Binarizing %s." % (input_filename))
        binarized_corpus = []
        for sentence,algin in zip(input_file,falign):
            if args.lowercase:
                print 'lowercase'
                sentence = sentence.lower()
            if args.char:
                print 'true'
                words = list(sentence.strip().decode('utf-8'))
            else:
                words = sentence.strip().split(' ')
                aligns = algin.strip().split(' ')
            binarized_sentence=[]
            if args.target:
                for i in range(len(words)):
                    ind=vocab.get(words[i])
                    if ind==None:
                        c1,e1=map(int,aligns[i].split(':'))
                        if abs(c1-e1)>7:
                            word='unkposn'
                        else:
                            word='unkpos_%i'%abs(e1-c1) if c1>e1 else 'unkpos%i'%abs(e1-c1)
                        ind = vocab.get(word)
                        # print c1, e1, word, ind
                        assert ind !=None
                    binarized_sentence.append(ind)
            else:
                binarized_sentence = [vocab.get(word, 1) for word in words]
            binarized_corpus.append(binarized_sentence)
        # endfor sentence in input_file
        # Output
        if args.each:
            if args.pickle:
                safe_pickle(binarized_corpus, base_filename + '.pkl')
        binarized_corpora += binarized_corpus
        input_file.seek(0)
    # endfor input_file in args.input
    if args.pickle:
        safe_pickle(binarized_corpora, args.binarized_text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    print type(args.target)
    print args.target
    base_filenames = open_files()
    combined_counter, sentence_counts, counters, vocab = create_dictionary()
    if args.ngram or args.pickle:
        binarize()
