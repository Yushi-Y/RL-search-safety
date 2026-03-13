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
        self._reset_penalty_stats()

    def _reset_penalty_stats(self):
        self._penalty_stats = {'penalties': [], 'projections': [], 'n_searches': 0, 'query_proj_pairs': []}

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

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

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)

            # Apply harmfulness penalty if enabled
            if self.harm_penalty is not None and self.harm_lambda > 0:
                pen_result = self.harm_penalty.compute_penalty(sequences_str)
                penalty = pen_result['penalty']
                score = score - self.harm_lambda * penalty
                # Collect stats for logging
                if pen_result['n_searches'] > 0:
                    self._penalty_stats['penalties'].append(penalty)
                    self._penalty_stats['projections'].extend(pen_result['projections'])
                    self._penalty_stats['n_searches'] += pen_result['n_searches']
                    # Collect (query, proj) pairs for top-k logging
                    for q, proj in zip(pen_result['queries'], pen_result['projections']):
                        self._penalty_stats['query_proj_pairs'].append((proj, q))

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # Log penalty summary for this batch
        if self.harm_penalty is not None and self._penalty_stats['n_searches'] > 0:
            projs = self._penalty_stats['projections']
            pens = self._penalty_stats['penalties']
            n_positive = sum(1 for p in projs if p > 0)
            print(f"[HarmPenalty] batch summary: {len(projs)} queries, "
                  f"{n_positive}/{len(projs)} positive proj, "
                  f"mean_proj={np.mean(projs):.2f}, "
                  f"mean_penalty={np.mean(pens):.4f}, "
                  f"max_proj={max(projs):.2f}")
            # Log top-10 highest projection queries
            top_pairs = sorted(self._penalty_stats['query_proj_pairs'], reverse=True)[:10]
            for proj, q in top_pairs:
                print(f"[HarmPenalty]   top proj={proj:.2f} query=\"{q[:120]}\"")
            self._reset_penalty_stats()

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

    # Set up harmfulness penalty if configured
    harm_penalty_obj = None
    harm_lambda = 0.0
    if getattr(config, 'harm_penalty', None) and config.harm_penalty.get('enable', False):
        # Expose GPUs to this process before any torch.cuda call
        harm_device = config.harm_penalty.get('device', 'cpu')
        if harm_device != 'cpu':
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            print(f"[HarmPenalty] Set CUDA_VISIBLE_DEVICES=0,1 for frozen model")
        from mitigation.harm_penalty import HarmPenalty
        harm_penalty_obj = HarmPenalty(
            direction_path=config.harm_penalty.direction_path,
            model_path=config.harm_penalty.model_path,
            layer_idx=config.harm_penalty.get('layer', 14),
            device=config.harm_penalty.get('device', 'cpu'),
        )
        harm_lambda = config.harm_penalty.get('lambda_coef', 0.02)
        print(f"[HarmPenalty] Enabled with lambda={harm_lambda}")

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
