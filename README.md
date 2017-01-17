# Gene Depth

This repository contains scripts for analyzing the depth of spots in background
regions.

## Segmentation

`segment.py segment input.tif output.tif` allows the segmentation of nD tif
files into distinct regions using a watershed method. The result can be
previewed with `segment.py plot output.tif`, and help on options can be
displayed with the `--help` argument.

## Depth

`depth.py calc` calculates the depth of a list of coordinates (see the
blob-detection repository) in a segmented image (see above). The depth can be
calculated in absolute or relative units, and displayed with `depth.py plot`.
