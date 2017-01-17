#!/usr/bin/env python3

from scipy.ndimage import morphology
import numpy as np
import pandas as pd
from tifffile import imread
from itertools import repeat
from pathlib import Path
from collections import defaultdict
from operator import itemgetter

def dual_distance(image, *args, **kwargs):
    distance = morphology.distance_transform_edt(image, *args, **kwargs)
    inverse_distance = morphology.distance_transform_edt(~image, *args, **kwargs)
    distance[~image] = -inverse_distance[~image]
    return distance


def load_spots(path):
    if path.suffix == ".pickle":
        with path.open("rb") as f:
            return load(f)
    elif path.suffix == ".csv":
        return np.loadtxt(str(path), ndmin=2, dtype='uint', delimiter=',')
    elif path.suffix == ".txt":
        return np.loadtxt(str(path), ndmin=2, dtype='uint')
    else:
        raise ValueError("Unrecognized file type: {}".format(path.suffix))

def calc(args):
    from sys import stdout

    segments = imread(str(args.segments))
    peaks = load_spots(args.spots)

    data = defaultdict(list)

    for segment in np.unique(segments)[1:]:
        background = segments == segment
        if args.dilation > 0:
            dilation = np.ones((args.dilation,) * segments.ndim)
            background = morphology.binary_dilation(background, dilation)
        distance = dual_distance(background, sampling=args.scale)

        cell_depths = distance[tuple(peaks[:, ::-1].T)]
        cell_depths = cell_depths[cell_depths >= 0.0]

        if len(cell_depths) != 1:
            continue

        data['gene'].extend([args.label] * len(cell_depths))
        if args.mode == 'absolute':
            data['depth'].extend(cell_depths)
        else:
            data['depth'].extend(cell_depths / distance.max() * 100)
    pd.DataFrame(data, columns=['gene', 'depth']).to_csv(stdout, index=False)


def plot(args):
    import matplotlib
    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(3, 6))

    data = pd.concat(map(pd.read_csv, map(str, args.depths)))
    ax.set_ylabel("Depth ({})".format(args.unit))
    grouped = data.groupby(['gene'], sort=False)
    keys, values = zip(*sorted(grouped, key=itemgetter(0)))
    labels = list(map("{}\n(n={})".format, keys, map(len, values)))
    ax.boxplot([v['depth'] for v in values], labels=labels)
    if args.unit == '%':
        ax.set_ylim([0, 100])
    plt.setp(ax.get_xticklabels(), rotation=90)

    fig.tight_layout()
    if args.outfile is not None:
        fig.savefig(str(args.outfile))
    else:
        plt.show()


def main(args=None):
    from argparse import ArgumentParser
    from sys import argv

    parser = ArgumentParser(description="Align two sets of points.")
    subparsers = parser.add_subparsers()

    calc_parser = subparsers.add_parser("calc")
    calc_parser.add_argument("label", type=str, help="The label to add to CSV rows")
    calc_parser.add_argument("segments", type=Path,
                             help="The segments to calculate depths in")
    calc_parser.add_argument("spots", type=Path,
                             help="The coordinates of peaks")
    calc_parser.add_argument("--mode", choices={'absolute', 'relative'}, default='absolute',
                             help="How to calculate the depth")
    calc_parser.add_argument("--dilation", type=int, default=0,
                             help="The amount to dilate each segment")
    calc_parser.add_argument("--scale", type=float, nargs='+',
                             help="The amount to dilate each segment")
    calc_parser.set_defaults(func=calc)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("depths", nargs='+', type=Path,
                             help="The depths to plot")
    plot_parser.add_argument("--unit", type=str, help="The unit for the y-axis")
    plot_parser.add_argument("--outfile", type=Path, help="Where to save the plot")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)


if __name__ == "__main__":
    main()
