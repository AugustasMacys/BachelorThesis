"""Training options."""

import argparse

parser = argparse.ArgumentParser(description="CoViAR")

# Model.
parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual'],
                    help='data representation.')
parser.add_argument('--arch', type=str, default="resnet152",
                    help='base architecture.')
parser.add_argument('--num_segments', type=int, default=3,
                    help='number of TSN segments.')
# Comeback
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')

# Training.
parser.add_argument('--epochs', default=50, type=int,
                    help='number of training epochs.')
parser.add_argument('--batch-size', default=12, type=int,
                    help='batch size.')
parser.add_argument('--lr', default=0.0003, type=float,
                    help='base learning rate.')
parser.add_argument('--lr-steps', default=[3, 6, 9], type=float, nargs="+",
                    help='epochs to decay learning rate.')
parser.add_argument('--lr-decay', default=0.5, type=float,
                    help='lr decay factor.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay.')

parser.add_argument('--workers', default=4, type=int,
                    help='number of data loader workers.')
