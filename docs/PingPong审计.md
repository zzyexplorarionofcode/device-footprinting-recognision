# PingPong 数据集审计（WAN 侧可行性）

## 结论

**PingPong 不是 WAN/NAT 后视角的数据集。** 它的 `remote` 指“远程手机控制设备”的场景，pcap 仍在家中 LAN 侧采集（私有 IP + 本地 MAC）。因此不能用它做“NAT 后外部攻击者能否看到长度特征”的外部验证。

## 包完整性

| 项 | 值 |
| --- | ---: |
| 文件 | `E:\dryad\PingPong (1).tar.gz` |
| 大小 | 32.67 GB |
| 条目 | 1,600 |
| pcap | 370 |
| gzip 头 | `1f 8b 08 00`，完整 |

## 结构

| 组 | pcap 数 | 说明 |
| --- | ---: | --- |
| local-phone | 244 | 本地手机控制，21 个设备 |
| ifttt | 68 | IFTTT 触发，9 个设备 |
| remote-phone | 36 | 远程手机控制，**仅 5 个设备** |
| public-dataset | 12 | blink-camera / tplink-plug / wemo |
| same-vendor | 10 | TP-Link 同厂商干扰项 |

pcap 命名：`<device>-<activity>.<iface>.<local|remote|detection|ifttt>.pcap`，`iface` 为 `eth0` 或 `wlan1`。

## 采样 pcap 审计

| pcap | 包数 | 时长 | 地址证据 | 端口 |
| --- | ---: | ---: | --- | --- |
| `rachio-sprinkler-quickrun.wlan1.remote.pcap` | 107,009 | 13,499 s | 192.168.1.154 / 192.168.1.1；本地 MAC | 443, 1900 |
| `rachio-sprinkler-quickrun.wlan1.local.pcap` | 270,284 | 13,399 s | 192.168.1.154 / 192.168.1.246 | 443 |
| `rachio-sprinkler-quickrun.wlan1.detection.pcap` | 553,452 | 15,640 s | 192.168.1.141 / 192.168.1.154 | 443, 56206 |

- link type 全部为 **Ethernet (1)**；
- `remote` 文件里仍是 `192.168.1.x` 私有地址与 `b0:b9:8a:...` 本地 MAC；
- 公网 IP（172.217.x、52.89.x）只是设备与云端的通信对端，**不是外部观测点**；
- `wlan1` 是本地无线监听接口，`eth0` 是有线接口，不是 WAN 口。

## 对研究的价值

| 用途 | 可行性 |
| --- | --- |
| WAN/NAT 后外部验证 | **不可行**：没有 WAN 侧 capture |
| 第二 LAN 部署的外部验证 | 有限可行：5 个 remote 设备、21 个 local 设备 |
| 多日 C→C/C→D/D→D | **不可行**：按活动/场景的短捕获，不是连续多日 |
| 负对照 | 已有：UNSW/YourThings/UNB 是公开数据副本 |

因此：PingPong 不能替代 Deakin/Mon 作为主数据集，也不能回答“NAT 后是否仍可识别”。若要做 WAN 验证，需要在路由器 WAN 口、ISP 侧或云侧采集，而不是这个包。

## 建议

1. 不投入适配器；保留作附录中的“第二 LAN 部署/远程控制场景”参考。
2. 若坚持 WAN 问题，另找 WAN-side pcap（例如在边界路由器镜像、云 VM 抓包）。
3. 论文中不要写“PingPong 提供 WAN capture”；只能写“remote-control scenario captured on LAN”。
