#!/usr/bin/env bash
gsutil -m cp gs://hprotein/code/*.py ./
gsutil -m cp gs://hprotein/code/*.sh ./
gsutil -m cp gs://hprotein/code/val_set.csv ./
chmod u+x copy_hprotein_from_gcs.sh
chmod u+x run_hprotein_gce.sh