#! /usr/bin/env bash

# Lints code:
#
#   # Lint mve by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh mve/{reporter,plot}.py

set -euo pipefail

lint() {
    PYTHONPATH=mve pylint --disable=locally-disabled,fixme,too-many-return-statements "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint mve
    else
        lint "$@"
    fi
}

main "$@"
