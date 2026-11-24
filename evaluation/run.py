"""Unified evaluation entry point; never starts training."""
import argparse
from utils import ROOT

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--root',default=str(ROOT));ap.add_argument('--diagnostic',action='store_true')
    ap.add_argument('--stage',choices=['all','audit','target_retrieval','generate','fem','fem_retrieval','statistics','report'],default='all');ap.add_argument('--retry-failed',action='store_true')
    args=ap.parse_args()
    from pipeline import execute
    execute(args)
