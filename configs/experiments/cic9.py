"""9 类 CIC 回退实验：LG Smart TV 在 20 天子集里只出现于 1 个抓包，无法按 capture 切分。

背景
----
官方 CIC-IoT-2022 的 Active 有 30 天；本机只有 20 天，且 ``LG Smart TV``（第 10 类）
只出现在 ``2021_11_03_Active.pcap`` 一个抓包里（454 条流）。``split_by_capture`` 要求
每个类别至少有两个独立抓包，否则无法同时进入训练和测试，因此 10 类主实验无法启动。

这是一个**明确标注的降级实验**，不是论文口径：
  * 数据：``flows_cic_active_9class.npz``（由 10 类 NPZ 按标签前缀派生，无重采样）；
  * 码本：``configs/methods/fade_cic9.json``（去掉 LG 对应的第 10 个码字，
    其余 9 个码字、预算、弱化器、抖动参数与 10 类配置逐字相同）；
  * 映射：old = 0..8，next = [7,3,8,1,5,4,2,0,6]（仍是 9 阶错排，无固定点）；
  * 输出：``artifacts/runs/fade_cic9/<case>``，与正式结果目录分开。

拿到更多 Active 天数、且 LG Smart TV 至少出现在两个抓包后，应回到
``configs/experiments/current.py`` 的 10 类 case，不要引用本文件的结果。
"""

from copy import deepcopy

from configs.experiments.current import EXPERIMENTS as FADE, ROOT

EXPERIMENTS = []

for original in deepcopy(FADE):
    if original["name"] != "cic_active_session_v2":
        continue
    case = deepcopy(original)
    case["name"] = "cic_active_9class"
    case["dataset"] = ROOT / "data/processed/cic_v2/flows_cic_active_9class.npz"
    case["expected_classes"] = 9
    case["output_dir"] = ROOT / "artifacts/runs/fade_cic9" / case["name"]
    # 方法名会进 CSV 分组键，必须与 10 类结果区分开。
    case["methods"][0]["name"] = "FADE-9class-old-to-next"
    case["methods"][0]["params"]["config"] = ROOT / "configs/methods/fade_cic9.json"
    EXPERIMENTS.append(case)
