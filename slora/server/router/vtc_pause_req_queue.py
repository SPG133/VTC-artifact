import time
import uuid
from collections import deque
from typing import List

from ..io_struct import Batch
from slora.server.router.req_queue import ReqQueue


class VTCPauseReqQueue(ReqQueue):
    """
    Minimal transplant of the two-condition pause policy.

    Important: 这里不是 LightLLM 那种完整的 paused/recover 状态机。
    当前版本通过“从 running batch 中移出 victim -> 重新入 waiting queue
    -> 后续重新 prefill”的方式近似实现抢占重算。
    """

    def __init__(
        self,
        max_total_tokens,
        batch_max_tokens,
        running_max_req_size,
        adapter_dirs,
        fair_weights,
        cost_func,
        victim_min_ratio_to_need,
    ) -> None:
        super().__init__(max_total_tokens, batch_max_tokens, running_max_req_size)
        self.adapter_dirs = adapter_dirs
        self.fair_weights = fair_weights
        self.cost_func = cost_func
        self.victim_min_ratio_to_need = victim_min_ratio_to_need

        self.served = {}
        self.user_req_list = {}

        self.avg_wait_ratio = 0.0
        self.finished_req_count = 0
        self.weighted_wait_ratio_sum = 0.0
        self.total_execution_weight = 0.0
        self.pause_req_list = []

    def append(self, req):
        self.waiting_req_list.append(req)
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.served[req.adapter_dir] = 0
        else:
            self.user_req_list[req.adapter_dir].append(req)

    def _req_size(self, req) -> int:
        # 这里用“输入长度 + 已生成长度”近似当前请求已经占用的 KV/上下文规模。
        return req.input_len + len(req.output_ids)

    def _estimate_standalone_latency(self, req) -> float:
        # 先用一个简单 proxy 近似独立完成时间，便于计算等待比例。
        return float(max(1, self._req_size(req) + (req.max_output_len - len(req.output_ids))))

    def _wait_ratio(self, req) -> float:
        if req.last_start_ts > 0:
            # 已经开始执行：等待时间 = 上次开始执行时刻 - 入队时刻。
            wait_time = max(0.0, req.last_start_ts - req.enqueue_ts)
        else:
            # 还没开始执行：等待时间 = 当前时刻 - 入队时刻。
            wait_time = max(0.0, time.time() - req.enqueue_ts)
        return wait_time / max(1.0, self._estimate_standalone_latency(req))

    def on_req_finished(self, req):
        # 请求完成后，用“等待比例 / 实际执行时间”更新系统平均等待比例。
        wait_ratio = self._wait_ratio(req)
        self.finished_req_count += 1
        self.weighted_wait_ratio_sum += wait_ratio * max(1e-6, req.last_execution_time)
        self.total_execution_weight += max(1e-6, req.last_execution_time)
        self.avg_wait_ratio = self.weighted_wait_ratio_sum / max(1e-6, self.total_execution_weight)

    def update_counter(self, current_batch: Batch):
        for req in current_batch.reqs:
            if req.has_generate_finished:
                self.on_req_finished(req)

    def _init_cache_list(self, current_batch: Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((self._req_size(req), req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    def _can_add_new_req(self, req, lora_ranks):
        import numpy as np

        cur_input_len = self._req_size(req)
        remain_output = req.max_output_len - len(req.output_ids)
        self.cache_len_list.append((cur_input_len + 1, remain_output - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num < self.max_total_tokens - self.adapter_size and len(self.cache_len_list) <= self.running_max_req_size

    def _select_running_victim(self, current_batch: Batch, need_token_num: int):
        if current_batch is None:
            return None
        min_need = need_token_num * max(1.0, float(self.victim_min_ratio_to_need))
        eligible = []
        for req in current_batch.reqs:
            cur_kv_like = self._req_size(req)
            # 你的两条规则：
            # 1. 只有显存规模大于当前请求需求 a 倍的强势任务，才允许被牺牲
            # 2. 该任务等待比例不能高于当前系统平均等待比例
            if cur_kv_like > min_need and self._wait_ratio(req) <= self.avg_wait_ratio:
                eligible.append(req)
        if not eligible:
            return None
        # 多个合格 victim 时，选择其中最小的强势任务，减少过度牺牲。
        eligible.sort(key=lambda req: self._req_size(req))
        return eligible[0]

    def pop_pause_reqs(self):
        reqs = self.pause_req_list
        self.pause_req_list = []
        return reqs

    def requeue_preempted_reqs(self, reqs: List):
        for req in reqs:
            if req.adapter_dir not in self.user_req_list:
                self.user_req_list[req.adapter_dir] = deque()
                self.served.setdefault(req.adapter_dir, 0)
            # 被抢占的任务回到 waiting queue 头部，后续重新 prefill 继续执行。
            self.user_req_list[req.adapter_dir].appendleft(req)
            self.waiting_req_list.insert(0, req)

    def generate_new_batch(self, current_batch: Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        if len(self.user_req_list) == 0:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        abort_list = []
        new_batch_total_tokens = 0
        active_served = {k: v for k, v in self.served.items()}

        while True:
            if len(active_served) == 0:
                break
            adapter_dir = min(active_served, key=active_served.get)
            if len(self.user_req_list[adapter_dir]) == 0:
                del active_served[adapter_dir]
                continue
            req = self.user_req_list[adapter_dir][0]
            if req.aborted:
                abort_list.append(req)
                self.user_req_list[adapter_dir].popleft()
                continue

            if self._can_add_new_req(req, lora_ranks) and new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                self.user_req_list[adapter_dir].popleft()
                active_served[adapter_dir] += req.input_len
                self.served[adapter_dir] += req.input_len
            else:
                victim = self._select_running_victim(current_batch, req.input_len)
                if victim is None:
                    del active_served[adapter_dir]
                    continue
                # 记录本轮要从 running batch 中移出的 victim。
                self.pause_req_list = [victim]
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                self.user_req_list[adapter_dir].popleft()
                active_served[adapter_dir] += req.input_len
                self.served[adapter_dir] += req.input_len
                break

        if len(can_run_list) == 0:
            return None
        new_batch = Batch(uuid.uuid4().hex, can_run_list)
        self.waiting_req_list = [req for req in self.waiting_req_list if req not in can_run_list and req not in abort_list]
        return new_batch
