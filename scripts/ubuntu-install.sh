#! /usr/bin/env bash
# Auto-install, using sudo, all system and python dependencies
# installs onto default pip. Should be called from repo root.
#
# ./scripts/ubuntu-install.sh

set -euo pipefail

pyv=$(python -c "import platform;print('.'.join(platform.python_version_tuple()[:2]))")
if [ "$pyv" != "3.5" ] ; then
    echo "python 3.5 exactly required" >&2
    exit 1
fi

distrib=$(lsb_release -a 2>/dev/null | grep "Distributor ID" | awk '{print $3}')
release=$(lsb_release -a 2>/dev/null | grep "Release" | awk '{print $2}')
if [ "$distrib" != "Ubuntu" ] ; then
    echo "expected Ubuntu, got $distrib $release" >&2
    exit 1
fi
if [ "$release" != "16.04" ] && [ "$release" != "14.04" ] ; then
    echo "expected Ubuntu 16.04 or 14.04, got $distrib $release" >&2
    exit 1
fi

# gym deps
sudo apt-get -qq update
sudo apt-get install -y \
     cmake zlib1g-dev libjpeg-dev xvfb libav-tools \
     xorg-dev libboost-all-dev swig libosmesa6-dev libglew-dev
sudo apt-get install -y libsdl2-dev || sudo apt-get install -f

pip install -r requirements.txt
