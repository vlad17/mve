"""
Adopted from HW3 of Deep RL.

Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10
random seeds. The runner code stored it in the directory structure

    data
    L expname_envname
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
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
import seaborn as sns


def _plot_data(data, value, outfile, hlines):
    sns.set(style='darkgrid', font_scale=1.5)
    sns.tsplot(data=data, time='iteration', value=value,
               unit='Unit', condition='Condition')
    for y, lbl in hlines:
        plt.axhline(y, label=lbl, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2)
    plt.savefig(outfile, format='pdf', bbox_inches='tight')
    plt.clf()


def _get_datasets(fpath, column, label, yaxis):
    unit = 0
    datasets = []
    for root, _, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            experiment_data = experiment_data[['iteration', column]]
            experiment_data.rename(columns={
                column: yaxis}, inplace=True)
            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
            )
            experiment_data.insert(
                len(experiment_data.columns),
                'Condition',
                label
            )

            datasets.append(experiment_data)
            unit += 1

    return datasets

def _get_hline(fpath, column, label):
    unit = 0
    vals = []
    for root, _, files in os.walk(fpath):
        if 'log.txt' in files:
            log_path = os.path.join(root, 'log.txt')
            experiment_data = pd.read_table(log_path)
            experiment_data.rename(columns={
                column: 'value'}, inplace=True)
            if len(experiment_data) == 1:
                top = experiment_data
            else:
                experiment_data = experiment_data[['iteration', 'value']]
                top = experiment_data.loc[
                    experiment_data['iteration'].idxmax()]
            vals.append(top['value'])
            unit += 1

    return np.mean(vals), label


def main():
    """Entry point for plot.py"""

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
    args = parser.parse_args()

    if not args.notex:
        matplotlib.rcParams['text.usetex'] = True

    data = []
    for column in args.columns:
        logdir, value, label = column.split(':')
        data += _get_datasets(logdir, value, label, args.yaxis)

    hlines = []
    for column in args.hlines:
        logdir, value, label = column.split(':')
        hlines.append(_get_hline(logdir, value, label))

    data = pd.concat(data, ignore_index=True)
    data = data[data['iteration'] >= args.drop_iterations]
    _plot_data(data, args.yaxis, args.outfile, hlines)


if __name__ == "__main__":
    main()
