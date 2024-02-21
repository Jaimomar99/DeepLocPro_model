# Held out predictions for all samples

This directory contains the held out predicted probabilities for all samples from nested cross-validation. In nested CV, each fold serves as the test set 4 times, leading to 5x4=20 test set results in total.

Files are named `model_{test_fold}_{validation_fold}.csv`.