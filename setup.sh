#!/usr/bin/env bash

cd ~
cp -r /opt/isaacgym ~/
python3 -m pip install -e ~/isaacgym/python
cd /tmp && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip && rm -f eigen-3.4.0.zip

python3 -m pip install 'pyglet<2'
python3 -m pip install --no-build-isolation -e ~/DyWA/dywa
git config --global --add safe.directory ~/DyWA
