"""
Adopted from HW3 of Deep RL.

Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10
random seeds. The runner code stored it in the directory structure

    data
    L expname_envname
      L  0
        L events.out.tfevents.*
        L params.json
      L  1
        L events.out.tfevents.*
        L params.json
       .
       .
       .
      L  9
        L events.out.tfevents.*
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py "data/expname_envname:AverageReturn:legend for this curve"
        --outfile x.pdf

To plot multiple items, try

    python plot.py "data/test1:AverageReturn:ar1" "data/test2:StdReturn:std2"
        --outfile y.pdf
"""

import argparse
import sys
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')
# flake8: noqa pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns


def plot_data(data, value, outfile, hlines, yrange):
    """
    Prints a sns tsplot to the outfile in PDF form
    """
    sns.set(style='darkgrid', font_scale=1.5)
    sns.tsplot(data=data, time='iteration', value=value,
               unit='Unit', condition='Condition')
    for y, lbl in hlines:
        plt.axhline(y, label=lbl, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2)
    if yrange:
        lo, hi = yrange
        plt.ylim(float(lo), float(hi))

    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.clf()


def get_datasets(fpath, column, label, yaxis, smoothing):
    """
    For a top-level directory containing multiple runs across several seeds,
    this gathers a pandas dataset with columns:

    iteration, Unit, Condition, yaxis

    Where iteration is the global training step for each run, the Unit
    is a unique index corresponding to a run with a single seed,
    the Condition is a constant column with value <label>, and
    the yaxis column contains the value in question (which is identified with
    the tag <column>)
    """

    unit = 0
    datasets = []
    for root, _, files in os.walk(fpath):
        for fname in files:
            if not fname.startswith('events.out.tfevents'):
                continue
            log_path = os.path.join(root, fname)
            df = []
            for e in tf.train.summary_iterator(log_path):
                for v in e.summary.value:
                    if v.tag == column:
                        df.append([e.step, unit, label, v.simple_value])
                        break
            columns = ['iteration', 'Unit', 'Condition', yaxis]
            experiment_data = pd.DataFrame(df, columns=columns)
            experiment_data[yaxis] = pd.rolling_mean(
                experiment_data[yaxis], smoothing)
            datasets.append(experiment_data)
            unit += 1
            break
    return datasets


def gather_data(args, yaxis, smoothing, drop_iterations):
    """
    Applies get_datasets to each argument in args, which should be a
    triplet (fpath, column, label), after appending the same yaxis
    and smoothing value.

    Then concatenates the resulting datasets, drops the specified
    number of initial iterations, and returns the final union-ed
    dataframe.
    """
    data = []
    for logdir, value, label in args:
        data += get_datasets(logdir, value, label, yaxis, smoothing)
    data = pd.concat(data, ignore_index=True)
    data = data[data['iteration'] >= drop_iterations]
    return data


def _main():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('columns', nargs='+')
    parser.add_argument('--outfile', default='', type=str, required=True)
    parser.add_argument('--yaxis', default='', type=str, required=True)
    parser.add_argument('--notex', default=False, action='store_true')
    parser.add_argument('--drop_iterations', default=0, type=int)
    # uses last record and plots it as a horizontal line
    parser.add_argument('--hlines', default=[], nargs='*', type=str)
    parser.add_argument('--smoothing', default=1, type=int)
    parser.add_argument('--yrange', default=None, type=float, nargs=2)
    args = parser.parse_args()

    if not args.notex:
        matplotlib.rcParams['text.usetex'] = True

    columns = [column.split(':') for column in args.columns]
    data = gather_data(columns, args.yaxis, args.smoothing,
                       args.drop_iterations)

    hlines = []
    for column in args.hlines:
        logdir, value, label = column.split(':')
        dfs = get_datasets(logdir, value, label, 'value', args.smoothing)
        values = []
        for df in dfs:
            top = df.loc[df['iteration'].idxmax()]
            values.append(top['value'])
        hlines.append([np.mean(values), label])

    plot_data(data, args.yaxis, args.outfile, hlines, args.yrange)


if __name__ == "__main__":
    _main()
