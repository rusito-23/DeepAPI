#!/bin/sh
# -- Download pre-trained InceptionNet weights -- # 

SRC_PATH=https://download.pytorch.org/models
SRC_NAME=googlenet-1378be20.pth

DST_PATH=./model
DST_NAME=googlenet_weights.pth

SRC=$SRC_PATH/$SRC_NAME
DST=$DST_PATH/$DST_NAME

[ ! -d $DST_PATH ] && mkdir -p $DST_PATH
wget $SRC -O $DST
