"""Derive exact leading-device subsets without resampling or reading PCAP again."""

# =============================================================================
# 中文阅读导图：从大数据集裁出“前 K 类”的严格子集
# =============================================================================
# 不重新读 PCAP、不重抽样，只按标签做布尔切片：
#   * keep = labels < classes，四数组、groups、capture_ids、partition_roles 同步切；
#   * device_names 取前 classes 个（设备顺序就是标签顺序，不能重排）；
#   * 在派生数据里写 derivation_manifest_json，记录父数据集路径/SHA-256/
#     操作说明/类别数/流数，并同步改写原 manifest 的 devices 与 audit 列表，
#     保证“子集是哪来的”可以追溯。
# 用途：论文里 4/6/8 类对比实验都从同一个父数据集派生，避免各自重新提取。
from pathlib import Path
import argparse
import json
import numpy as np
from .pcap import load_dataset, save_dataset, validate_dataset
from experiment_support.provenance import sha256_file

def derive_prefix(source: Path, output: Path, classes: int):
    source,output=source.resolve(),output.resolve()
    if source==output:raise ValueError('subset output must differ from source')
    if output.exists():raise FileExistsError(output)
    data=load_dataset(source)
    names=np.asarray(data['device_names'])
    if not 1<=classes<len(names):raise ValueError('subset class count must be between 1 and source class count - 1')
    n=len(data['labels']);keep=np.asarray(data['labels'])<classes
    subset={key:(value[keep].copy() if key!='device_names' and value.ndim and len(value)==n else value.copy()) for key,value in data.items()}
    subset['device_names']=names[:classes].copy()
    derivation={'parent_dataset':str(source),'parent_sha256':sha256_file(source),'operation':'exact leading label subset; no resampling','class_count':classes,'flows':int(keep.sum())}
    subset['derivation_manifest_json']=np.array(json.dumps(derivation,ensure_ascii=False))
    for key in ('unsw_extraction_manifest_json','extraction_manifest_json'):
        if key not in subset:continue
        manifest=json.loads(str(subset[key]))
        manifest.update(devices=names[:classes].tolist(),class_count=classes,derivation=derivation)
        if 'four_class_devices' in manifest:manifest['four_class_devices']=names[:min(4,classes)].tolist()
        if 'device_audits' in manifest:manifest['device_audits']=[row for row in manifest['device_audits'] if row['device'] in names[:classes]]
        subset[key]=np.array(json.dumps(manifest,ensure_ascii=False))
    validate_dataset(subset)
    save_dataset(output,subset)
    return derivation

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--classes',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(derive_prefix(args.source,args.output,args.classes),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
