#! /usr/bin/env bash

# This is the complementary teardown script to scripts/launch-ray-aws.sh
# If the above script was used to create a cluster with name NAME then
#
# ./scripts/teardown-ray-aws.sh NAME
#
# will tear down that cluster and delete the corresponding file in ~/ray-clusters

if [ $# -ne 1 ] ; then
    echo 'usage: ./scripts/teardown-ray-aws.sh cluster-name'
    exit 1
fi

set -euo pipefail

CLUSTER_NAME="$1"
CLUSTER_FILE="$HOME/ray-clusters/${CLUSTER_NAME}.yaml"

if ! [ -f "$CLUSTER_FILE" ] ; then
    echo "cluster $CLUSTER_NAME not found in ~/ray-clusters"
    exit 1
fi

yes | ray teardown "$CLUSTER_FILE" || true

rm -f "$CLUSTER_FILE"
rm -f "$HOME/ray-clusters/${CLUSTER_NAME}.ssh"
rm -f "$HOME/ray-clusters/${CLUSTER_NAME}.scp"
rm -f "$HOME/ray-clusters/${CLUSTER_NAME}.tunnel"
