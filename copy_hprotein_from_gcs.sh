#!/usr/bin/env bash
gsutil -m cp gs://hprotein/code/*.py ./
gsutil -m cp gs://hprotein/code/*.sh ./
gsutil -m cp gs://hprotein/code/*.csv ./
chmod u+x copy_hprotein_from_gcs.sh
chmod u+x run_hprotein_gce.sh
chmod u+x run_ensemble_gce.sh
chmod u+x run_per_model_ensemble_gce.sh