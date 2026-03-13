# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., harm_penalty=None, harm_lambda=0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.harm_penalty = harm_penalty  # HarmPenalty instance or None
        self.harm_lambda = harm_lambda

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # --- First pass: decode all sequences and compute task scores ---
        all_sequences_str = []
        all_scores = []
        all_valid_response_lengths = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            all_sequences_str.append(sequences_str)
            all_scores.append(score)
            all_valid_response_lengths.append(valid_response_length)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # --- Batched harm penalty (single forward pass) ---
        if self.harm_penalty is not None and self.harm_lambda > 0:
            import time as _time
            n_search = sum(1 for s in all_sequences_str if '<search>' in s)
            print(f"[HarmPenalty] Computing penalty for {len(all_sequences_str)} sequences ({n_search} with <search>)...")
            _t0 = _time.time()
            pen_result = self.harm_penalty.compute_penalty_batch(all_sequences_str)
            print(f"[HarmPenalty] Done in {_time.time() - _t0:.1f}s")
            for i in range(len(data)):
                if pen_result['has_search'][i]:
                    all_scores[i] -= self.harm_lambda * pen_result['penalties'][i]

            # Log penalty summary
            search_indices = [i for i, h in enumerate(pen_result['has_search']) if h]
            if search_indices:
                projs = [pen_result['projections'][i] for i in search_indices]
                pens = [pen_result['penalties'][i] for i in search_indices]
                n_positive = sum(1 for p in projs if p > 0)
                print(f"[HarmPenalty] batch summary: {len(projs)} queries, "
                      f"{n_positive}/{len(projs)} positive proj, "
                      f"mean_proj={np.mean(projs):.2f}, "
                      f"mean_penalty={np.mean(pens):.4f}, "
                      f"max_proj={max(projs):.2f}")
                # Log top-10 highest projection queries
                top_pairs = sorted(
                    [(pen_result['projections'][i], all_sequences_str[i]) for i in search_indices],
                    reverse=True,
                )[:10]
                import re as _re
                for proj, q in top_pairs:
                    # Extract first generated search query (skip prompt template)
                    matches = _re.findall(r'<search>(.*?)</search>', q, _re.DOTALL)
                    q_text = matches[1].strip()[:120] if len(matches) > 1 else (matches[0].strip()[:120] if matches else q[:120])
                    print(f"[HarmPenalty]   top proj={proj:.2f} query=\"{q_text}\"")

        # --- Write scores into reward tensor ---
        for i in range(len(data)):
            reward_tensor[i, all_valid_response_lengths[i] - 1] = all_scores[i]

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # Set up harmfulness penalty if configured (CPU by default to avoid stealing GPUs from RL)
    harm_penalty_obj = None
    harm_lambda = 0.0
    if getattr(config, 'harm_penalty', None) and config.harm_penalty.get('enable', False):
        from mitigation.harm_penalty import HarmPenalty
        harm_device = config.harm_penalty.get('device', 'cpu')
        harm_penalty_obj = HarmPenalty(
            direction_path=config.harm_penalty.direction_path,
            model_path=config.harm_penalty.model_path,
            layer_idx=config.harm_penalty.get('layer', 14),
            device=harm_device,
        )
        harm_lambda = config.harm_penalty.get('lambda_coef', 0.02)
        print(f"[HarmPenalty] Enabled on {harm_device} with lambda={harm_lambda}")

    reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=0,
        harm_penalty=harm_penalty_obj, harm_lambda=harm_lambda,
    )

    # Note that we always use function-based RM for validation (no penalty)
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
