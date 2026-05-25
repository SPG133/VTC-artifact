# VTC Pause Policy Port

## 1. 这是什么

这份修改把你在 LightLLM 版本里定义的“两条件 pause 策略”移植到了 VTC/S-LoRA 的调度框架中，并新增了一个 scheduler：

- `vtc_pause`

## 2. 重要说明

这不是把 LightLLM 中现成的 running-request `pause/recover` 生命周期完整搬到 VTC 上。

原因是：

- LightLLM 原版已经有 running request 的 pause / recover 机制
- VTC / S-LoRA 当前这套 runtime 更偏 queue-based batching 和 fairness scheduling
- 它没有同等成熟的“中途打断一个 running request 并在后续恢复”的现成路径

因此，这次移植做的是：

> 把你的策略先落成一个 **memory-aware admission / batch scheduling policy**

而不是：

> 完整的 running-request pause / active KV swap 系统

## 3. 移植后的两条规则

当前 `vtc_pause` 只保留两条规则。

### 规则 1：强势任务门槛

存在一个当前 running request 满足：

```text
victim_size > a * current_request_need
```

其中：

- `current_request_need` 当前近似用新请求的 `input_len`
- `victim_size` 当前近似用 running request 的 `input_len + generated_len`
- `a` 由参数 `--victim-min-ratio-to-need` 控制
- 默认值是 `5.0`

### 规则 2：等待比例门槛

候选 running request 还必须满足：

```text
victim.wait_ratio <= avg_wait_ratio
```

其中：

- `wait_ratio = waiting_time / estimated_standalone_latency`
- `estimated_standalone_latency ≈ input_len + max_output_len`
- `avg_wait_ratio` 用已完成请求的等待比例按执行时间加权维护

## 4. 当前在 VTC 上的真实语义

这版 `vtc_pause` 的真实含义是：

- 当一个新请求准备进入 batch 时
- 调度器会检查当前 running batch 中是否存在“允许让位”的强势任务
- 只有存在这样的大任务时，才允许新请求按当前策略被接纳

所以它更像：

```text
memory-aware admission policy
```

而不是：

```text
true running-request pause / recover
```

## 5. 改动位置

### 新 scheduler

- [slora/server/router/vtc_pause_req_queue.py](./slora/server/router/vtc_pause_req_queue.py)

### scheduler 注册

- [slora/server/router/manager.py](./slora/server/router/manager.py)

新增名称：

- `vtc_pause`

### 参数入口

- [slora/server/api_server.py](./slora/server/api_server.py)
- [slora/server/input_params.py](./slora/server/input_params.py)

新增参数：

- `--victim-min-ratio-to-need`

### request 状态

- [slora/server/io_struct.py](./slora/server/io_struct.py)

新增字段：

- `enqueue_ts`
- `last_start_ts`
- `finish_ts`
- `last_execution_time`
- `total_wait_time`

## 6. 如何运行

在根目录安装：

```bash
cd /home/lz/桌面/LightLLM-main/VTC-artifact-pause
pip install -e .
```

启动：

```bash
cd fair_bench
python launch_server.py --scheduler vtc_pause
```

如果要指定倍率：

```bash
python launch_server.py --scheduler vtc_pause --victim-min-ratio-to-need 5.0
```

## 7. 这版适合做什么

适合：

- 验证“强势任务门槛 + 平均等待比约束”这套策略思想
- 在 VTC 的公平调度实验框架下快速比较不同 `a` 的效果
- 观察短任务 tail latency 是否改善

不适合：

- 宣称已经实现完整 running pause
- 宣称已经实现 active KV swap
- 直接等价于 LightLLM 版本中真正的 pause/recover 语义

## 8. 下一步

如果要继续逼近真正的 running pause，需要额外补：

- running batch 中 request 的中断能力
- victim 从当前 batch 中移出后的 batch 重构
- victim 后续恢复路径

这部分复杂度明显高于当前这版最小策略移植。 
