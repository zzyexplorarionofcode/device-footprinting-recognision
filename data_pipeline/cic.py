"""Versioned CIC TCP/TLS session extraction, with whole-PCAP capture identities.

No defense-dependent sample filter is used. Each PCAP is checkpointed separately;
existing shards are reused only when input identity and the extraction protocol match.
"""

# =============================================================================
# 中文阅读导图：CIC-IoT-2022 的“设备目录 → 会话 NPZ”
# =============================================================================
# 这是全包最短的一条端到端数据链，适合第一次理解“抓包如何变成 [N,32]”。
# 本文件只做三件事：把同一 TCP 连接的包收进 Session、按规则断开、编码成定宽流。
#
# 关键对象：
#   Session        一条会话：最多 width 个包的前缀、总包数、是否见过 TLS、
#                  首个 SYN 身份（direction+seq）、双向 FIN 集合；
#   Sessionizer    active = {(设备label, 双向五元组): Session}。
#                  新 SYN 且序列不同 → 视为新连接；RST 或双向 FIN → 立即结束；
#                  另有 expire(now) 清理超时会话；
#   finish()       只保留“至少 min_packets 包 + 出现 TLS 记录 + 至少一个上行包”
#                  的会话，然后调用 pcap._encode_flow 编码；
#   prepare()      按 capture 逐文件处理，每个 PCAP 一个 shard + 一个 json 报告，
#                  报告里记录输入文件大小/mtime/protocol 与 shard 的 SHA-256；
#                  重启时只有“输入没变且协议没变”才复用 shard。
#
# 为什么按整份 PCAP 分组：同一抓包里的时间邻近流高度相关，若按流随机切分会泄漏。
# 因此 group 存的是 path.name，后续 split_by_capture 以它为单位切分。
#
# 代码风格提示：本文件刻意使用紧凑写法（分号、短变量名）以贴近原始提取脚本，
# 但语义与 pcap.py 完全一致：第一包 gap=0、mask 左对齐、方向 ±1。
#
# 复现命令：python -B -m data_pipeline.cic --root <数据根> --mode active
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.pcap import ParsedPacket, read_pcap, parse_packet, _encode_flow, validate_dataset

PROTOCOL='cic-tcp-sessions-v2-first-packet-origin'
ROOT=Path(__file__).resolve().parents[1]
DEVICES={
 'active':['Amazon Alexa Echo Dot 1','Amazon Alexa Echo Dot 2','Amazon Alexa Echo Spot',
           'Amazon Alexa Echo Studio','Arlo Base Station','Arlo Q Camera','Sonos One Speaker',
           'Netatmo Camera','Smart Board','LG Smart TV'],
 'idle':['Amazon Alexa Echo Dot 1','Amazon Alexa Echo Dot 2','Amazon Alexa Echo Studio',
         'Amazon Alexa Echo Spot','Arlo Base Station','Sonos One Speaker','Philips Hue Bridge',
         'Google Nest Mini','Smart Board','Teckin Plug 1']}

@dataclass
class Session:
    prefix: list[ParsedPacket]=field(default_factory=list)
    count: int=0
    tls: bool=False
    last: float=0.0
    syn_identity: tuple | None=None
    fins: set=field(default_factory=set)

    def add(self,p,width):
        self.count+=1; self.tls |= p.is_tls_record; self.last=p.timestamp
        if len(self.prefix)<width:self.prefix.append(p)
        if p.tcp_flags & 2 and not p.tcp_flags & 16:self.syn_identity=(p.direction,p.tcp_sequence)
        if p.tcp_flags & 1:self.fins.add(p.direction)

class Sessionizer:
    def __init__(self,timeout=120.0,width=32):
        if not np.isfinite(timeout) or timeout<=0:raise ValueError('timeout must be positive')
        self.timeout,self.width=timeout,width
        self.active={}

    def push(self,key,p):
        completed=[]; current=self.active.get(key)
        initial_syn=bool(p.tcp_flags & 2 and not p.tcp_flags & 16)
        new_syn=initial_syn and current is not None and current.syn_identity != (p.direction,p.tcp_sequence)
        if current is not None and (p.timestamp-current.last>self.timeout or p.timestamp<current.last or new_syn):
            completed.append(current);current=None
        if current is None:current=Session();self.active[key]=current
        current.add(p,self.width)
        if p.tcp_flags & 4 or len(current.fins)==2:
            completed.append(current);del self.active[key]
        return completed

    def expire(self,now):
        keys=[k for k,s in self.active.items() if now-s.last>self.timeout]
        return [(k,self.active.pop(k)) for k in keys]

def device_macs(path,names):
    import openpyxl
    workbook=openpyxl.load_workbook(path,read_only=True,data_only=True)
    try:
        mapping={str(row[1]):bytes.fromhex(str(row[2]).replace(':','').replace('-',''))
                 for row in workbook.active.iter_rows(values_only=True)
                 if len(row)>2 and row[1] in names and row[2]}
    finally:workbook.close()
    if set(names)-mapping.keys():raise ValueError('missing device MACs: '+str(set(names)-mapping.keys()))
    if any(len(m)!=6 for m in mapping.values()) or len(set(mapping.values()))!=len(names):
        raise ValueError('device MACs must be unique six-byte identifiers')
    return mapping

def extract_capture(path,macs,*,width=32,min_packets=4,timeout=120.0):
    names=list(macs); lookup={mac:i for i,mac in enumerate(macs.values())}
    sessions=Sessionizer(timeout,width);samples=[];counts={'records':0,'sessions':0,'accepted':0}
    last_expire=None
    def finish(label,session):
        counts['sessions']+=1
        if session.count<min_packets or not session.tls:return
        if not any(p.direction==1 for p in session.prefix):return
        samples.append(_encode_flow(session.prefix,label=label,group=path.name,max_packets=width))
        counts['accepted']+=1
    for record in read_pcap(path):
        counts['records']+=1;frame=record.frame
        if len(frame)<14:continue
        # A selected-device-to-selected-device frame belongs to both endpoint observations.
        selected=set(lookup[m] for m in (frame[6:12],frame[:6]) if m in lookup)
        for label in selected:
            p=parse_packet(record.timestamp,frame,macs[names[label]],443,frame_id=record.frame_id)
            if p is None or p.flow_key[0]!=6 or 443 not in (p.source_port,p.destination_port):continue
            for finished in sessions.push((label,p.flow_key),p):finish(label,finished)
        if last_expire is None:last_expire=record.timestamp
        if record.timestamp-last_expire>timeout:
            for key,session in sessions.expire(record.timestamp):finish(key[0],session)
            last_expire=record.timestamp
    for key,session in sessions.active.items():finish(key[0],session)
    fields=('lengths','timestamps','frame_ids','source_ips','destination_ips','gaps','directions','mask')
    data={k:np.stack([getattr(s,k) for s in samples]) for k in fields} if samples else {}
    data.update(labels=np.array([s.label for s in samples],dtype=np.int16),
                groups=np.array([s.group for s in samples],dtype=str))
    return data,counts

def atomic_npz(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**data)
    temp.replace(path)

def prepare(root,mode,output,*,limit=None):
    names=DEVICES[mode];macs=device_macs(root/'Device List.xlsx',names)
    source=root/('active' if mode=='active' else '2-Idle')
    files=sorted(source.glob('*.pcap'))
    if not files:raise FileNotFoundError(source)
    if limit is not None:files=files[:limit]
    shard_dir=output.parent/'shards'/mode; shard_dir.mkdir(parents=True,exist_ok=True)
    protocol=dict(id=PROTOCOL,width=32,min_packets=4,tls_port=443,timeout_seconds=120.0,
                  devices=names,macs={k:v.hex() for k,v in macs.items()},
                  session_boundaries=['new_SYN_sequence','RST','both_FIN','120s_idle','capture_end'],
                  first_packet_wait_observable=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('pcap.py').read_bytes()).hexdigest())
    shards=[];reports=[]
    for ordinal,path in enumerate(files):
        identity=dict(path=str(path.resolve()),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        shard=shard_dir/(path.stem+'.npz');meta=shard.with_suffix('.json')
        expected=dict(input=identity,protocol=protocol)
        if meta.exists() and shard.exists():
            report=json.loads(meta.read_text(encoding='utf-8'))
            if any(report.get(k)!=v for k,v in expected.items()):raise ValueError(f'stale extraction shard: {shard}')
            if report.get('shard_sha256')!=hashlib.sha256(shard.read_bytes()).hexdigest():raise ValueError(f'corrupt shard: {shard}')
        else:
            print(f'[{mode} {ordinal+1}/{len(files)}] {path.name}: extracting',flush=True)
            started=time.monotonic();data,counts=extract_capture(path,macs)
            atomic_npz(shard,data)
            report=dict(**expected,counts=counts,seconds=time.monotonic()-started,
                        shard_sha256=hashlib.sha256(shard.read_bytes()).hexdigest())
            meta.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[{mode} {ordinal+1}/{len(files)}] {path.name}: {report["counts"]["accepted"]} sessions',flush=True)
        shards.append(shard);reports.append(report)
    parts=[]
    for shard in shards:
        with np.load(shard,allow_pickle=False) as z:
            if len(z['labels']):parts.append({k:z[k] for k in z.files})
    if not parts:raise ValueError('no sessions extracted')
    combined={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    combined['device_names']=np.asarray(names)
    if limit is not None:
        present=np.unique(combined['labels'])
        combined['labels']=np.searchsorted(present,combined['labels']).astype(np.int16)
        combined['device_names']=np.asarray(names)[present]
    combined['capture_ids']=combined['groups'].copy()
    combined['data_protocol']=np.array(PROTOCOL)
    combined['extraction_manifest_json']=np.array(json.dumps(dict(protocol=protocol,captures=reports,partial=limit is not None),ensure_ascii=False))
    validate_dataset(combined,expected_width=32)
    atomic_npz(output,combined)
    print(f'Saved {output}: {len(combined["labels"])} sessions',flush=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('F:/iot dataset/cic2022'))
    parser.add_argument('--mode',choices=DEVICES,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit-captures',type=int,help='diagnostic only; writes a separate sample file')
    args=parser.parse_args(argv)
    suffix='_sample' if args.limit_captures else ''
    output=args.output or ROOT/f'data/processed/cic_v2/flows_cic_{args.mode}_10class{suffix}.npz'
    prepare(args.root,args.mode,output,limit=args.limit_captures)

if __name__=='__main__':main()
