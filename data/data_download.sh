#!/bin/bash

mkdir DIV2k
cd DIV2k

# Training.
echo "Downloading DIV2k train dataset."
wget http://data.vision.ee.ethz.ch/cvl/DIV2k/DIV2k_train_HR.zip
unzip DIV2k_train_HR.zip
mv DIV2k_train_HR train
rm DIV2k_train_HR.zip

# Validating.
echo "Downloading DIV2k valid dataset."
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2k_valid_HR.zip
mv DIV2k_valid_HR valid
rm DIV2k_valid_HR.zip
