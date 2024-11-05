#!/usr/bin/env python3
import scipy.stats as stats
import numpy as np
from stats import decideColumn
from stats import extractData

def main(operators_fname, obs1_name, obs2_name, warmup):
  colX_num = decideColumn(operators_fname, obs1_name)
  colY_num = decideColumn(operators_fname, obs2_name)

  warmupX, colX = extractData(operators_fname, colX_num, warmup)
  warmupY, colY = extractData(operators_fname, colY_num, warmup)

  return (stats.pearsonr(colX, colY))

if __name__ == '__main__':
 import argparse
 parser = argparse.ArgumentParser()
 parser.add_argument('-f', '--fname',default='./operators.dat',type=argparse.FileType('r'),help="Operator output file")
 parser.add_argument('-o', '--observables',nargs=2,type=str,help='observable names')
 parser.add_argument('-w', '--warmup',default=100,type=int)

 args = parser.parse_args()

 pearson_coeff, _ = main(args.fname, args.observables[0], args.observables[1], args.warmup)
 print(pearson_coeff)
