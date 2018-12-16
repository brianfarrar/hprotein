#!/usr/bin/env bash
gsutil -m cp *.py gs://hprotein/code
gsutil -m cp *.sh gs://hprotein/code
gsutil -m cp val_set.csv gs://hprotein/code