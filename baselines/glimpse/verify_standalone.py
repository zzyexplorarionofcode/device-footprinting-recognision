"""复制本目录到临时位置，并在新解释器中禁止导入旧版及其他项目模块。"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    source = Path(__file__).resolve().parent
    child = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

bundle = Path(sys.argv[1]).resolve()
class RejectExternalProjectCode(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in {'glimpse_seq64_oracle', 'our_method', 'defense_baselines',
                    'universal_experiment', 'dataset_pcap', 'method32', 'method32_tools'}:
            raise AssertionError('Forbidden external project import: ' + fullname)
        return None

# 某些嵌入式Python的_pth文件含旧方法目录，删除这类搜索路径并独立阻止导入。
sys.path[:] = [p for p in sys.path if 'glimpse-seq64-oracle' not in p.lower()]
sys.path.insert(0, str(bundle))
sys.meta_path.insert(0, RejectExternalProjectCode())
sys.argv = [str(bundle / 'self_test.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
import demo
import glimpse_seq64_oracle_v2
assert Path(demo.__file__).resolve().is_relative_to(bundle)
assert Path(glimpse_seq64_oracle_v2.__file__).resolve().is_relative_to(bundle)
print('Independent relocated package, training and save/load checks passed.')
'''
    with tempfile.TemporaryDirectory(prefix="glimpse_v2_standalone_") as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        for name in ("demo.py", "self_test.py", "requirements.txt"):
            shutil.copy2(source / name, bundle / name)
        shutil.copytree(
            source / "glimpse_seq64_oracle_v2", bundle / "glimpse_seq64_oracle_v2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        subprocess.run([sys.executable, "-I", "-c", child, str(bundle)], check=True, cwd=bundle)


if __name__ == "__main__":
    main()
