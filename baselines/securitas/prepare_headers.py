"""Prepare verified PCAP header sidecars locally before moving runs to Linux."""
from pathlib import Path
import argparse
import numpy as np
from securitas_adapted.config import load_config
from securitas_adapted.data import prepare_data,recover_headers

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--capture-root',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--partition',choices=('train-validation','test'),required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new sidecar path; existing headers are not overwritten')
    config=load_config(args.config)
    data,train,val,split=prepare_data(config)
    indices=np.sort(np.r_[train,val]) if args.partition=='train-validation' else np.asarray(split['outer_test_indices'])
    recover_headers(data,indices,args.output,split['dataset_sha256'],args.capture_root,print)
    print(f'Verified {len(indices)} sessions; saved {args.output}',flush=True)

if __name__=='__main__': main()
