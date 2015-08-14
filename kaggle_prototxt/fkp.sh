#!/usr/bin/env sh

./build/tools/caffe train -solver examples/kaggle_prototxt/fkp_solver.prototxt -gpu 0
