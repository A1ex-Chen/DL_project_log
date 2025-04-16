#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
import os

NO_RELATION = "NA"



if __name__ == "__main__":
    # Parse the arguments from stdin
    #dir = 'output_tacred_new'
    #dir = 'output_fewrel_new'
    dir = sys.argv[1]
    data_dir = os.listdir(dir)
    data_dir_gold = {file.strip().split("_")[2]:file for file in data_dir if "test_gold_" in file}
    data_dir_pred = {file.strip().split("_")[2]:file for file in data_dir if "test_pred_" in file}
    for id,gold in data_dir_gold.items():
        #args = parse_arguments(dir+'/'+gold,dir+'/'+data_dir_pred[id])
        #args = [ dir+'/'+gold, dir+'/'+ data_dir_pred[id] ]
        gold_file = dir + '/' + gold
        pred_file = dir + '/'+ data_dir_pred[id]

        #key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
        #prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]
        key = [str(line).rstrip('\n') for line in open(str(gold_file))]
        prediction = [str(line).rstrip('\n') for line in open(str(pred_file))]

        # Check that the lengths match
        if len(prediction) != len(key):
            print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
            exit(1)

        # Score the predictions
        #score(key, prediction, verbose=True)

        fout = open(dir+"/test_paper_results_"+id,'w')
        prec_micro, recall_micro, f1_micro = score(key, prediction, verbose=True)
        fout.write("{}\n".format(id))
        fout.write( "Precision (micro): {:.3%}\n".format(prec_micro) )
        fout.write( "   Recall (micro): {:.3%}\n".format(recall_micro) )
        fout.write( "       F1 (micro): {:.3%}\n".format(f1_micro) )
        fout.write("=============================================================\n")
        fout.close()
