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
import seaborn as sns


def _plot_data(data, value, outfile):
    sns.set(style='darkgrid', font_scale=1.5)
    sns.tsplot(data=data, time='aggregation iteration', value=value,
               unit='Unit', condition='Condition')
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
            experiment_data = experiment_data[['Iteration', column]]
            experiment_data.rename(columns={
                'Iteration': 'aggregation iteration',
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
    args = parser.parse_args()

    if not args.notex:
        matplotlib.rcParams['text.usetex'] = True

    data = []
    for column in args.columns:
        logdir, value, label = column.split(':')
        data += _get_datasets(logdir, value, label, args.yaxis)

    data = pd.concat(data, ignore_index=True)
    data = data[data['aggregation iteration'] >= args.drop_iterations]
    _plot_data(data, args.yaxis, args.outfile)


if __name__ == "__main__":
    main()
