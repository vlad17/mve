#! /usr/bin/env bash

# Lints code:
#
#   # Lint cmpc by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh cmpc/{reporter,plot}.py

set -euo pipefail

lint() {
    PYTHONPATH=cmpc pylint --disable=locally-disabled,fixme,too-many-return-statements "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint cmpc
    else
        lint "$@"
    fi
}

main "$@"
