"""
Ever wonder what it looks like when a grenade goes off in a hyperparameter
factory? Tranining a (potentially bootstrapped) on-policy MPC controller
involves a _lot_ of hyperparameters that we'd like to toggle from the command
line. This file helps do that without going crazy:

    import flags
    all_flags = flags.get_all_flags()
"""

import argparse
import collections

def convert_flags_to_json(flags):
    """
    Returns a "jsonnable" dict representation of Flags iterable. For example,

    Parameters
    ----------
    flags: Flags iterable

    Returns
    -------
    dict
    """
    params = dict()
    for flag in flags:
        params[flag.name()] = vars(flag)
    return params

class Flags(object):
    """A group of logically related flags."""

    def __str__(self):
        xs = vars(self)
        return "\n".join(["--{} {}".format(x, v) for (x, v) in xs.items()])

    def __repr__(self):
        return str(self)

    def name(self):
        """Flag group name"""
        return self.__class__.__name__.replace('Flags', '').lower()

def parse_args(flags):
    """
    Imagine we had the following Flags:

        class A(Flags):
            @staticmethod
            def add_flags(parser):
                a = parser.add_argument_group('a')
                a.add_argument('--a')

            @staticmethod
            def name():
                return "A"

            def __init__(self, args):
                self.a = args.a

        class B(Flags):
            @staticmethod
            def add_flags(parser):
                b = parser.add_argument_group('b')
                b.add_argument('--b')

            @staticmethod
            def name():
                return "B"

            def __init__(self, args):
                self.b = args.b

    `parse_args([A, B])` parses all the flags registered by `A.add_flags` and
    `B.add_flags`. It then returns a namedtuple `{"A": A(args), "B": B(args)}`
    where `args` are the parsed flags.

        args = parse_args([A, B])
        print(args.A.a)
        print(args.B.b)
    """
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    for flag in flags:
        flag.add_flags(parser)

    args = parser.parse_args()
    kwargs = {flag.name(): flag(args) for flag in flags}
    namedtuple = collections.namedtuple("Args", kwargs.keys())
    return namedtuple(**kwargs)


def parse_args_with_subcmds(flags, subflags):
    """
    Look at the documentation for parse_args. Consider the same Flags classes A
    and B. Imagine we have similar one named C. `parse_args_with_subcmds([A],
    [B, C])` will create a command line parser which has `B` and `C` as
    subcommands. The `B` subcommand will take the A and B flags; the `C`
    subcommand will take the A and C subcommands:

        python prog.py B --a --b     # ok
        python prog.py C --a --c     # ok
        python prog.py B --a --b --c # not ok; no flag c for subcommand B

    `parse_args_with_subcmds` will return a tuple (args, subflag) where `args`
    is a namedtuple of all the flags (as returned by `parse_args`) and
    `subflag` is the subflag object that corresponds to the chosen subcommand.

        python prog.py B --a --b # returns (args, B(args))
        python prog.py C --a --c # returns (args, C(args))
    """
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest="subcommand")

    for subflag in subflags:
        subparser = subparsers.add_parser(subflag.name(),
                                          formatter_class=formatter_class)
        for flag in flags:
            flag.add_flags(subparser)
        subflag.add_flags(subparser)

    args = parser.parse_args()
    subflags_by_name = {subflag.name(): subflag for subflag in subflags}
    if args.subcommand is None:
        names = list(subflags_by_name.keys())
        msg = "No subcommand chosen. Choose one of {}.".format(names)
        raise ValueError(msg)
    chosen_subflag = subflags_by_name[args.subcommand]
    kwargs = {flag.name(): flag(args) for flag in flags + [chosen_subflag]}
    namedtuple = collections.namedtuple("Args", kwargs.keys())
    t = namedtuple(**kwargs)
    return t, t._asdict()[args.subcommand]
