"""
Plot the dynamics MSE from open-loop predictions as a function of the
horizon of prediction.

Averages across multiple seeds if the additional data is present.

See plot.py documentation for additional data directory documentation.
"""

import argparse
import sys

import matplotlib
matplotlib.use('Agg')
# flake8: noqa pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

from plot import gather_data


def _main():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datadirs', nargs='+')
    parser.add_argument('--outfile', default='', type=str, required=True)
    parser.add_argument('--notex', default=False, action='store_true')
    parser.add_argument('--hsteps', nargs='+', type=int)
    parser.add_argument('--smoothing', default=100, type=int)
    parser.add_argument('--yrange', default=None, type=float, nargs=2)
    parser.add_argument('--time', default=None, type=int,
                        help='at which point during training to plot the '
                        'smoothed error, by default the latest available '
                        'to all seeds')
    args = parser.parse_args()

    if not args.notex:
        matplotlib.rcParams['text.usetex'] = True

    fmt = str(max(len(str(i)) for i in args.hsteps))
    fmt = 'dynamics/open loop/{:' + fmt + 'd}-step mse'
    runs = {}
    yaxis = 'MA({}) dynamics MSE'.format(args.smoothing)
    for datadir_name in args.datadirs:
        datadir, name = datadir_name.split(':')
        columns = []
        for i in args.hsteps:
            col_name = fmt.format(i)
            columns.append((datadir, col_name, i))
        runs[name] = gather_data(columns, yaxis, args.smoothing, 0)

    if args.time is None:
        args.time = min(
            df[['Unit', 'iteration']].groupby('Unit').max()['iteration'].min()
            for df in runs.values())

    ixs = {}
    mus = {}
    tops = {}
    bots = {}
    for name, df in runs.items():
        df = df[df['iteration'] == args.time].rename(columns={
            'Unit': 'seed', 'Condition': 'step'})[[
                'step', yaxis]]
        mu = df.groupby('step').mean()
        ixs[name] = mu.index
        mu = mu[yaxis]
        mus[name] = mu
        tops[name] = df.groupby('step').quantile(0.95)[yaxis]
        bots[name] = df.groupby('step').quantile(0.05)[yaxis]

    for name in runs:
        plt.plot(ixs[name], mus[name], label=name)
        plt.fill_between(ixs[name], bots[name], tops[name], alpha=0.3)

    if args.yrange:
        lo, hi = args.yrange
        plt.ylim(float(lo), float(hi))

    plt.xlabel('horizon')
    plt.ylabel(yaxis)
    plt.title('open-loop dynamics (at episode {})'.format(args.time))
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc=2)
    plt.savefig(args.outfile, format='pdf', bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    _main()
