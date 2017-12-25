"""
This file defines the general interface to adding flags for configuration
and processing the generated flags.

A Flags instance should correspond to a logically grouped set of options
for a program.

To create a set of flags, construct
a class as follows inheriting from Flags:

class MyFlags(Flags):
    def __init__(self, ...):
        super().__init__('myflags', 'my flags pretty name', arguments)

The constructor can take whatever arguments it pleases, but the super class
requires a name (which should be unique across the whole program and
should be a valid python variable name), a pretty name that can be any string,
and a list of arguments, each of which is an ArgSpec.

In the main file, invoke parse_args(flags) with the list of flags to be
parsed from sys.argv. The returned object will be an instance x, where
x.myflags will be the MyFlags instance, which will AUTOMATICALLY have
attributes corresponding to the names of its arguments.

If one of the arguments is named 'arg1', then it should be
correspondingly invoked in the command line as:

python main.py --arg1 value_for_arg1
"""

import argparse
import shlex
import sys


class ArgSpec:
    """
    Specification for a single flag.
    The name should be a valid python variable name.

    All the keywords should correspond to argparse argument-creation
    keywords.
    """

    def __init__(self, name='', **kwargs):
        self.name = name
        self.kwargs = kwargs


class Flags(object):
    """
    A group of logically related flags. Extend this class to create
    a set of flags that is automatically parsed.

    The Flags class shouldn't be used directly externally otherwise.

    A Flags instance corresponds to an argparse argument group.
    """

    def __init__(self, name, pretty_name, arguments):
        self.name = name
        self._pretty_name = pretty_name
        self._args = arguments

    def add_to_parser(self, parser):
        """
        Adds own arguments (prefixed by Flags name) to parser
        in its own argument group.
        """
        arg_grp = parser.add_argument_group(self._pretty_name)
        for arg in self._args:
            arg_grp.add_argument(
                '--' + arg.name,
                **arg.kwargs)

    def set_from_parsed(self, parsed_args):
        """
        Given (all) parsed arguments, set own corresponding attributes
        appropriately.
        """
        for arg in self._args:
            argval = getattr(parsed_args, arg.name)
            setattr(self, arg.name, argval)

    def values(self):
        """
        Returns a dictionary of the presumed-set values.
        """
        return {arg.name: getattr(self, arg.name) for arg in self._args}


def parse_args(flags, cmdargs=None):
    """
    Please see the module documentation.

    If cmdargs is not None, then this list is used for parsing arguments
    instead of sys.argv.

    In the case of subflags being None, all Flags instances in the
    flags list are parsed from the current command-line arguments or filled
    with their defaults.

    A top-level flags object is returned, with attributes corresponding
    to flag names. Consider the following example:

    class A(Flags):
        def __init__(self):
            super().__init__('a', 'a', [ArgSpec('arga', int, 0, 'help')])

    class B(Flags):
        def __init__(self):
            super().__init__('b', 'b', [ArgSpec('argb', int, 0, 'help')])

    x = parse_args([A(), B()])

    These flags can be set with

        python main.py --arga 1 --argb 3

    And in the code they may be accessed as x.a.arga or x.b.argb. It's up
    to the user to make sure the names 'a' and 'b' don't clash, and that
    the argument names 'arga' and 'argb' don't clash.

    If cmdargs is not None, then parse that tokenized list instead.
    """
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    for flag in flags:
        flag.add_to_parser(parser)

    args = parser.parse_args(args=cmdargs)
    for flag in flags:
        flag.set_from_parsed(args)

    cmdargs = cmdargs if cmdargs is not None else sys.argv
    cmdargs = [sys.executable] + cmdargs[:]
    invocation = ' '.join(shlex.quote(s) for s in cmdargs)

    return _TopLevelFlags(flags, invocation)


class _TopLevelFlags:
    def __init__(self, flags, cmd):
        self._cmd = cmd
        self._flags = flags
        for flag in flags:
            setattr(self, flag.name, flag)

    def asdict(self):
        """
        A serializable dictionary version of the invoked flags.
        """
        all_flags = {}
        for flag in self._flags:
            all_flags[flag.name] = flag.values()
        all_flags['invocation'] = self._cmd
        return all_flags
