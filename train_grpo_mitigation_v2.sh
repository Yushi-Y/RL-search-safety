export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='data/nq_search_mixed'

WAND_PROJECT='Search-R1'

export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
export EXPERIMENT_NAME=search-r1-grpo-qwen2.5-7b-it-em-mitigation-v2

DIRECTION_PATH='interp_results/directions/search_query_direction_qwen7b/search_query_direction.json'
DIRECTION_LAYER=14
HARM_LAMBDA=8.0

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray}
mkdir -p $RAY_TMPDIR

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=32 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=500 \
    data.max_start_length=600 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.test_freq=-1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=101 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    +harm_penalty.enable=true \
    +harm_penalty.direction_path=$DIRECTION_PATH \
    +harm_penalty.model_path=$BASE_MODEL \
    +harm_penalty.layer=$DIRECTION_LAYER \
    +harm_penalty.lambda_coef=$HARM_LAMBDA \
    +harm_penalty.device="cpu" \
    +harm_penalty.use_relu=false \
    2>&1 | tee "${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
