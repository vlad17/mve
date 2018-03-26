#! /usr/bin/env bash

# Easy script to unzip and install mujoco, assuming that there's an mjkey.txt
# in the current direcotry. Installs mujoco 1.50 in the expected location in a
# subfolder of home, moving the key into the mujoco folder.

set -eo pipefail

if ! [ -d ~/.mujoco/mjpro150 ] ; then
    rm -f mjpro150_linux.zip
    wget -q https://www.roboti.us/download/mjpro150_linux.zip -O mjpro150_linux.zip
    mkdir -p ~/.mujoco
    unzip mjpro150_linux.zip -d ~/.mujoco
    rm mjpro150_linux.zip
fi

if [ -f mjkey.txt ] ; then
    mkdir -p ~/.mujoco
    mv mjkey.txt ~/.mujoco
fi

if [ -f ~/.mujoco/mjkey.txt ] ; then
    echo 'WARNING: no ~/.mujoco/mjkey.txt found, please add manually before continuing'
fi

