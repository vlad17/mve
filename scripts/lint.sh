#! /usr/bin/env bash

# Lints code:
#
#   # Lint mpc_bootstrap by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh mpc_bootstrap/{logz,plot}.py

set -euo pipefail

lint() {
    PYTHONPATH=mpc_bootstrap pylint --disable=locally-disabled,fixme "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint mpc_bootstrap
    else
        lint "$@"
    fi
}

main "$@"
