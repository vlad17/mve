#!/bin/bash
# Wraps passed-in command with a fake display so that rendering can work on a remote server

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- "$@"
