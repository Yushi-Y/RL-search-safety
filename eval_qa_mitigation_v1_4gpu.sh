#!/bin/bash
# Evaluate QA (EM) on saved VERL checkpoints using val_only mode.
# Usage: bash eval_checkpoints.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/nq_search'
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_BASE="verl_checkpoints/search-r1-grpo-qwen2.5-7b-it-em-mitigation/actor"
CHECKPOINTS=("global_step_25" "global_step_50" "global_step_75" "global_step_100")

# Also eval the base IT model as step 0
ALL_MODELS=("${CHECKPOINTS[@]/#/$CKPT_BASE/}")

for MODEL_PATH in "${ALL_MODELS[@]}"; do
    STEP_NAME=$(basename "$MODEL_PATH")
    echo "============================================"
    echo "Evaluating: $MODEL_PATH ($STEP_NAME)"
    echo "============================================"

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=32 \
        data.val_batch_size=32 \
        data.max_prompt_length=1024 \
        data.max_response_length=500 \
        data.max_start_length=600 \
        data.max_obs_length=600 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.use_kl_loss=true \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=true \
        actor_rollout_ref.actor.fsdp_config.grad_offload=true \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        algorithm.no_think_rl=false \
        actor_rollout_ref.rollout.n_agent=5 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=true \
        trainer.logger=['console'] \
        +trainer.val_only=true \
        +trainer.val_before_train=true \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=25 \
        trainer.test_freq=-1 \
        trainer.project_name=Search-R1 \
        trainer.experiment_name=eval-mitigation-$STEP_NAME \
        trainer.total_epochs=15 \
        trainer.total_training_steps=100 \
        trainer.default_local_dir=verl_checkpoints/eval-mitigation-$STEP_NAME \
        max_turns=4 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        2>&1 | tee eval-mitigation-$STEP_NAME.log

    echo ""
done

# Collect results from log files into a single JSON
python3 -c "
import re, json

results = {}
models = [
('global_step_25', 'eval-mitigation-global_step_25.log', 25),
    ('global_step_50', 'eval-mitigation-global_step_50.log', 50),
    ('global_step_75', 'eval-mitigation-global_step_75.log', 75),
    ('global_step_100', 'eval-mitigation-global_step_100.log', 100),
]
for name, logfile, step in models:
    try:
        with open(logfile) as f:
            text = f.read()
        # Match val/test_score/nq or similar keys with float values
        scores = re.findall(r\"'(val/test_score/\w+)':\s*([\d.]+)\", text)
        entry = {'step': step, 'model': name}
        for key, val in scores:
            entry[key] = float(val)
        results[name] = entry
    except FileNotFoundError:
        results[name] = {'step': step, 'model': name, 'error': 'log not found'}

with open('eval_results_mitigation.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print('Saved to eval_results_mitigation.json')
"
