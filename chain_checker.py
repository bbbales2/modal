#!/usr/bin/env python

import sys
import numpy
import matplotlib.pyplot as plt
import argparse
import re

parser = argparse.ArgumentParser(prog='Chain Checker')
parser.add_argument('--prefix', default = "", nargs='?', help='Discard all lines in files except those with this prefix')
parser.add_argument('files', nargs='+', help='Files with samples in them')

args = parser.parse_args()

def parse(fname):
    samples = {}
    
    with open(fname) as f:
        for line in f:
            if len(args.prefix) > 0 and not re.match(args.prefix, line):
                continue
            
            tokens = line.strip().split(',')

            for key, val in zip(tokens[0::2], tokens[1::2]):
                key = key.strip()
                val = float(val)
                
                if key not in samples:
                    samples[key] = []

                samples[key].append(val)

    for key in samples:
        samples[key] = numpy.array(samples[key])

    return samples

chains = []
for fname in args.files:
    chains.append(parse(fname))

ckeys = [set(chain.keys()) for chain in chains]

keys = set.intersection(*ckeys)

if ckeys[0] != keys:
    raise Exception("Files appear to have differently labeled parameters, or different ones altogether")

if len(ckeys[0]) == 0:
    raise Exception("At least one file appears to have no keys")

keysList = sorted(list(keys))

print " " * 9 + " ".join("{0:<15}".format(key) for key in keysList)
for i, chain in enumerate(chains):
    print "Chain {0}: ".format(i) + " ".join("{0:<+5.3f} ({1:<+4.3f})".format(numpy.mean(chain[key]), numpy.std(chain[key])) for key in keysList)

def check_mixing(chains):
    minl = min(len(chain) for chain in chains)

    chainsPostWarmup = [chain[-minl // 2:] for chain in chains]

    n = (minl // 2) // 2

    chains = []
    for chain in chainsPostWarmup:
        chains.append(chain[:n])
        chains.append(chain[-n:])

    m = len(chains)
        
    chainMeans = numpy.mean(chains, axis = 1)

    meanTotal = numpy.mean(chainMeans)

    B = n * sum((chainMeans - meanTotal)**2) / (m - 1)

    chainVars = []
    for chain, chainMean in zip(chains, chainMeans):
        chainVars.append(sum((chain - chainMean)**2) / (len(chain) - 1))

    W = sum(chainVars) / m

    var = (n - 1) * W / n + B / n

    return numpy.sqrt(var / W), W, B

for key in keysList:
    tchains = [chain[key] for chain in chains]

    R, W, B = check_mixing(tchains)

    print "Parameter {0:<5} Rhat {1:<5} Within variance {2:<5} Between variance {3:<5}".format(key, R, W, B)
    
