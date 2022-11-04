#!/bin/bash

# Movielens
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip

mkdir ./data
mv ml-25m.zip
pushd ./data
unzip ml-25m.zip
popd