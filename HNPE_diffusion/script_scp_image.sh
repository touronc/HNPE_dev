#!/bin/bash

SOURCE_DIR="ctouron@access.grid5000.fr:grenoble/HNPE_diff/logs"
DEST_DIR="./results/compar"
for i in {15..24}; do
    FOLDER="$SOURCE_DIR/$i/artifacts/image"
    scp "$FOLDER"/* "$DEST_DIR"/
    FOLDER="$SOURCE_DIR/$i/metrics"
    scp "$FOLDER"/* "$DEST_DIR"/
done
echo "Done!"