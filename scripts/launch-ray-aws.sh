#! /usr/bin/env bash

# Generates a ray cluster with the current branch of the repo checked out.
# Should be called from from repository root.
# arguments are cluster name, aws machine type, tasks per node, maximum number of workers
# and then arguments to be invoked by the tune file, if any.
#
# I.e., the following:
#
# ./scripts/launch-ray-aws.sh gpu-cluster g3.8xlarge 4 2
#
# creates (or updates) a cluster called 'gpu-cluster' with g3.8xlarge workers
# where each workers will be multiplexed over 4 tasks (so 2 tasks per GPU).
#
# Keep in mind:
# * cluster names should be unique (and live clusters are tied to files in $HOME/ray-clusters)
# * don't mess with files managed by this script in $HOME/ray-clusters
# * don't run two versions of this script at once
# * this script assumes your AWS credentials are in ~/.aws and the corresponding profile
#   has sufficient permissions to create this cluster, etc.

if [ $# -ne 4 ] ; then
    echo 'usage: ./scripts/launch-ray-aws.sh cluster-name aws-machine-type tasks-per-machine max-workers'
    exit 1
fi

set -euo pipefail

CLUSTER="$1"
shift
AWS_MACHINE_TYPE="$1"
shift
TASKS_PER_MACHINE="$1"
shift
MAX_WORKERS="$1"
shift

HEAD_SHA=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
ORIGIN_SHA=$(git rev-parse origin/$BRANCH)

if [ "$HEAD_SHA" != "$ORIGIN_SHA" ] ; then
    echo "HEAD sha != origin sha for branch $BRANCH -- did you git push?"
    exit 1
fi

mkdir -p $HOME/ray-clusters
CLUSTER_YAML="$HOME/ray-clusters/$CLUSTER.yaml"
OUTFILE="/tmp/$CLUSTER-create.out"

echo "launching cluster $CLUSTER with sha = $HEAD_SHA"
echo "tee to $OUTFILE for output"

cat ./scripts/ray-config-template.yaml | \
    sed 's|<<<HOME>>>|'"$HOME"'|' | \
    sed 's|<<<AWS_MACHINE_TYPE>>>|'"$AWS_MACHINE_TYPE"'|' | \
    sed 's|<<<TASKS_PER_MACHINE>>>|'"$TASKS_PER_MACHINE"'|' | \
    sed 's|<<<HEAD_SHA>>>|'"$HEAD_SHA"'|' | \
    sed 's|<<<MAX_WORKERS>>>|'"$MAX_WORKERS"'|' | \
    sed 's|<<<CLUSTER>>>|'"$CLUSTER"'|' > $CLUSTER_YAML

yes | ray create_or_update $CLUSTER_YAML | tee $OUTFILE || true

SSH_CMD=$(tail -2 $OUTFILE)
SSHFILE="$HOME/ray-clusters/$CLUSTER.ssh"
echo "$SSH_CMD" > $SSHFILE

echo "cluster launched - ssh cmd in $SSHFILE"

