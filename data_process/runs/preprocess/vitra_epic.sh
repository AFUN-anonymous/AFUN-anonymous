#!/usr/bin/env bash
# Preprocess VITRA EPIC-KITCHENS.
# Episodes without a hand-object contact in epic_contact_index are skipped.
set -euo pipefail

python preprocess.py vitra_epic \
    multi_processing=true \
    num_processes=20
