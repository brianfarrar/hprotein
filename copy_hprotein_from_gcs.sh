#!/usr/bin/env bash
gsutil -m cp gs://hprotein/code/*.py ./
gsutil -m cp gs://hprotein/code/*.sh ./
chmod u+x copy_hprotein_from_gcs.sh
chmod u+x run_hprotein_gce.sh