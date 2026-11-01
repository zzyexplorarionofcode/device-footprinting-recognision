"""Physical timing metrics independent of the observer's first-packet origin."""

# =============================================================================
# 中文阅读导图：三个时间指标的统一口径（全包最短但最容易被误解的文件）
# =============================================================================
# 观察者没有“首包之前”的共享时钟，所以特征里首包 gap 恒为 0；但物理开销必须
# 以原始首包为时间零点计算，否则会漏掉“首包被延迟”的那部分成本。
#
# timing_metrics(source_gaps, event_times, is_real)：
#   source_end = 原始流所有 gap 之和（原始完成时刻）
#   real_end   = 最后一个“真实”事件的时刻        → real_completion_delay_seconds
#   wire_end   = 最后一个“发送”事件的时刻        → wire_completion_delay_seconds
#   二者之差                                    → cover_tail_seconds（尾部覆盖）
# 三个值都 max(0, ...)，只报告“变慢”，不会出现负开销。
#
# store_timing(stats, index, values, size)：把结果写进逐流 stats 字典；
# 键不存在时按需分配数组，因此新旧调用方可以共用同一张成本表。
import numpy as np
TIMING_KEYS=('real_completion_delay_seconds','wire_completion_delay_seconds','cover_tail_seconds')
def timing_metrics(source_gaps,event_times,is_real):
    """Event timestamps use the ORIGINAL first packet as time zero, including added waits."""
    gaps=np.asarray(source_gaps,dtype=float)
    source_end=float(gaps[1:].sum()) if len(gaps) else 0.0
    times=np.asarray(event_times,dtype=float);real=np.asarray(is_real,dtype=bool)
    real_end=float(times[real].max()) if np.any(real) else source_end
    wire_end=float(times.max()) if len(times) else source_end
    return dict(real_completion_delay_seconds=max(0.,real_end-source_end),
                wire_completion_delay_seconds=max(0.,wire_end-source_end),
                cover_tail_seconds=max(0.,wire_end-real_end))
def store_timing(stats,index,values,size):
    for key,value in values.items():
        if key not in stats:stats[key]=np.zeros(size,dtype=float)
        stats[key][index]=value
