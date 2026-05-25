import time
import uuid
from collections import deque
from typing import List

from ..io_struct import Batch
from slora.server.router.req_queue import ReqQueue


class VTCPauseReqQueue(ReqQueue):
    """
    Minimal transplant of the two-condition pause policy.

    Important: VTC/S-LoRA does not currently expose the same running-request
    pause/recover lifecycle as LightLLM. This queue therefore applies the
    policy at admission / batch construction time instead of true mid-run KV
    preemption.
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

    def append(self, req):
        self.waiting_req_list.append(req)
        if req.adapter_dir not in self.user_req_list:
            self.user_req_list[req.adapter_dir] = deque([req])
            self.served[req.adapter_dir] = 0
        else:
            self.user_req_list[req.adapter_dir].append(req)

    def _estimate_standalone_latency(self, req) -> float:
        return float(max(1, req.input_len + req.max_output_len))

    def _wait_ratio(self, req) -> float:
        if req.last_start_ts > 0:
            wait_time = max(0.0, req.last_start_ts - req.enqueue_ts)
        else:
            wait_time = max(0.0, time.time() - req.enqueue_ts)
        return wait_time / max(1.0, self._estimate_standalone_latency(req))

    def on_req_finished(self, req):
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
                self.cache_len_list.append((req.input_len + len(req.output_ids), req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    def _can_add_new_req(self, req, lora_ranks):
        import numpy as np

        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
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

    def _eligible_dominant_running_req_exists(self, current_batch: Batch, need_token_num: int) -> bool:
        if current_batch is None:
            return True
        min_need = need_token_num * max(1.0, float(self.victim_min_ratio_to_need))
        eligible = []
        for req in current_batch.reqs:
            cur_kv_like = req.input_len + len(req.output_ids)
            if cur_kv_like > min_need and self._wait_ratio(req) <= self.avg_wait_ratio:
                eligible.append(req)
        if not eligible:
            return False
        eligible.sort(key=lambda req: req.input_len + len(req.output_ids))
        return True

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

            if not self._eligible_dominant_running_req_exists(current_batch, req.input_len):
                del active_served[adapter_dir]
                continue

            if self._can_add_new_req(req, lora_ranks) and new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
                self.user_req_list[adapter_dir].popleft()
                active_served[adapter_dir] += req.input_len
                self.served[adapter_dir] += req.input_len
            else:
                break

        if len(can_run_list) == 0:
            return None
        new_batch = Batch(uuid.uuid4().hex, can_run_list)
        self.waiting_req_list = [req for req in self.waiting_req_list if req not in can_run_list and req not in abort_list]
        return new_batch
