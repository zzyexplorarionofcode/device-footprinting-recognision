"""Atomic, checksummed artifacts for explicitly resumed trusted local runs."""

# =============================================================================
# 中文阅读导图：断点续跑的“可信本地”实现
# =============================================================================
# 设计目标：长实验中途崩溃/被终止后，不必重算已经完成的策略与攻击者，
# 但绝不能把“另一份配置/另一份数据”的权重混进来。
#
# run_guard(output)
#     目录级互斥锁：在 <output>/.run.guard 上取非阻塞文件锁（Windows 用 msvcrt，
#     POSIX 用 fcntl）。进程被 SIGKILL 时操作系统会自动释放锁，所以锁文件本身
#     永远保留（不删除 inode），只做加锁/解锁。
#
# Checkpoints(output, contract, resume=False)
#     contract 是调用方算出的“语义指纹”（配置+数据+源码+软件版本）。
#     index.json 里保存 contract 与每个条目的文件名/SHA-256/耗时。
#     * 首次运行：创建 checkpoints/，写 index；
#     * resume=True 但目录里没有 index：报错（旧版运行不能自动续跑）；
#     * contract 不一致：报错，提示新建运行目录；
#     * load(key)：校验路径必须在本目录内且 SHA-256 匹配，防止手改/串目录；
#     * save(key)：临时文件 → joblib 压缩 → fsync → os.replace 原子改名 → 更新 index；
#     * get(key, compute)：命中则打印“[复用]”，否则计算、保存、记录耗时。
#
# 典型用途：一次实验里 policy/<方法名>、query_model/<模型名>、
# application/<repeat>/<方法>/<target>、result/... 都是独立 key。
import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import joblib
from experiment_support.provenance import atomic_json, sha256_file


@contextmanager
def run_guard(output):
    """OS releases this lock even after SIGKILL; keep the inode in place."""
    handle=open(Path(output)/'.run.guard','a+b')
    locked=False
    try:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as exc:
            raise FileExistsError('该运行目录正被另一个进程使用') from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()


class Checkpoints:
    def __init__(self, output, contract, *, resume=False):
        self.root = Path(output)/'checkpoints'
        self.index_path = self.root/'index.json'
        self.contract = contract
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding='utf-8'))
            if self.index['contract'] != contract:
                raise ValueError('恢复被拒绝：数据、配置、代码或软件版本不同；请创建新运行目录')
        else:
            if resume:
                raise ValueError('此目录没有可恢复检查点；旧版运行不能自动续跑')
            self.root.mkdir(parents=True, exist_ok=True)
            self.index = {'schema':1, 'contract':contract, 'entries':{}}
            atomic_json(self.index_path,self.index)

    def has(self, key):
        return key in self.index['entries']

    def load(self, key):
        entry=self.index['entries'][key]
        path=self.root/entry['file']
        if path.parent.resolve()!=self.root.resolve() or sha256_file(path)!=entry['sha256']:
            raise ValueError(f'检查点损坏，拒绝复用：{key}')
        return joblib.load(path)

    def save(self, key, value):
        fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=self.root)
        os.close(fd)
        try:
            joblib.dump(value,tmp,compress=3)
            with open(tmp,'r+b') as f: os.fsync(f.fileno())
            name=hashlib.sha256(key.encode()).hexdigest()+'-'+sha256_file(tmp)+'.joblib'
            destination=self.root/name
            os.replace(tmp,destination)
            self.index['entries'][key]={'file':name,'sha256':sha256_file(destination)}
            atomic_json(self.index_path,self.index)
        finally:
            Path(tmp).unlink(missing_ok=True)
        return value

    def get(self, key, compute):
        if self.has(key):
            print(f'[复用] {key}',flush=True)
            return self.load(key)
        print(f'[计算] {key}',flush=True)
        started=time.perf_counter()
        atomic_json(self.root.parent/'current_task.json',{'task':key,'status':'running','started_at':time.time()})
        value=compute()
        self.save(key,value)
        self.index['entries'][key]['seconds']=time.perf_counter()-started
        atomic_json(self.index_path,self.index)
        atomic_json(self.root.parent/'current_task.json',{'task':key,'status':'saved',
                    'seconds':self.index['entries'][key]['seconds']})
        print(f'[已保存] {key}',flush=True)
        return value
