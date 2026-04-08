# VData Space Release Guide

When `/VData` is full, use this guide to free up space from the shared HuggingFace cache at `/VData/resources/huggingface/hub/`.

## Quick commands

```bash
# Check available space
df -h /VData

# List models sorted by size
du -sh /VData/resources/huggingface/hub/models--*/ | sort -rh | head -20

# Delete a model
rm -rf /VData/resources/huggingface/hub/models--<org>--<model>/
```

## Already deleted

| Model | Size | Last Modified | Deleted On |
|---|---|---|---|
| `meta-llama/Llama-2-70b-chat-hf` | 129 GB | 2025-05-22 | 2026-04-07 |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 132 GB | 2025-03-18 | 2026-04-07 |
| `meta-llama/Meta-Llama-3-70B` | 132 GB | 2025-07-06 | 2026-04-07 |

## Priority 1: Large + old (safe to delete)

These are 100GB+ models last modified 6+ months ago. Delete these first.

| Model | Size | Last Modified | Last Accessed |
|---|---|---|---|
| `deepseek-ai/DeepSeek-R1` | 642 GB | 2025-03-25 | >1 year ago |
| `meta-llama/Meta-Llama-3-70B-Instruct` | 132 GB | 2025-04-21 | ~1 year ago |
| `meta-llama/Llama-3.1-70B-Instruct` | 132 GB | 2025-05-16 | 11 months ago |
| `nvidia/Llama-3.3-Nemotron-70B-Reward-Multilingual` | 132 GB | 2025-08-04 | 8 months ago |
| `Qwen/Qwen2.5-Math-72B` | 136 GB | 2025-08-21 | 8 months ago |
| `Qwen/Qwen2.5-Math-72B-Instruct` | 136 GB | 2025-08-21 | 8 months ago |

**Subtotal: ~1,310 GB**

## Priority 2: Large + moderately old

These are 100GB+ models, 4-6 months old.

| Model | Size | Last Modified | Last Accessed |
|---|---|---|---|
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | 203 GB | 2025-05-06 | 11 months ago |
| `meta-llama/Llama-4-Scout-17B-16E` | 203 GB | 2025-12-06 | 4 months ago |
| `meta-llama/Llama-3.1-70B` | 132 GB | 2025-09-27 | 6 months ago |
| `Qwen/Qwen2.5-72B-Instruct` | 136 GB | 2025-09-25 | 6 months ago |
| `Qwen/Qwen2.5-72B` | 136 GB | 2025-09-24 | 6 months ago |
| `Qwen/Qwen3-Next-80B-A3B-Thinking` | 152 GB | 2025-11-23 | 5 months ago |
| `openai/gpt-oss-120b` | 122 GB | 2025-10-08 | 6 months ago |
| `PeterJinGo/SearchR1-...-qwen2.5-32b-...-v0.3` | 123 GB | 2025-07-29 | 8 months ago |

**Subtotal: ~1,207 GB**

## Priority 3: Medium models, old

| Model | Size | Last Modified | Last Accessed |
|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 62 GB | 2025-03-11 | >1 year ago |
| `Qwen/Qwen2.5-32B-Instruct` | 62 GB | 2025-04-08 | ~1 year ago |
| `Qwen/Qwen2.5-32B` | 62 GB | 2025-04-11 | ~1 year ago |
| `Qwen/Qwen2.5-Coder-32B` | 62 GB | 2025-09-24 | 6 months ago |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 62 GB | 2025-09-24 | 6 months ago |
| `meta-llama/Llama-2-13b-chat-hf` | 25 GB | 2025-11-26 | 4 months ago |
| `meta-llama/Llama-2-7b-chat-hf` | 13 GB | 2025-11-26 | 4 months ago |
| `meta-llama/Llama-2-7b-hf` | 13 GB | 2025-05-22 | 11 months ago |

**Subtotal: ~361 GB**

## DO NOT delete (actively used by kebl6672)

| Model | Reason |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | Base model for ARL training |
| `Qwen/Qwen2.5-7B` | Used in experiments |
| `Qwen/Qwen2.5-14B-Instruct` | Used in experiments |
| `PeterJinGo/SearchR1-*-7b-*` | SearchR1 baselines |
| `PeterJinGo/SearchR1-*-14b-*` | SearchR1 baselines |

## Also check your own space

```bash
# Your verl checkpoints (568 GB as of 2026-04-07)
du -sh /VData/kebl6672/ARL/verl_checkpoints/*/

# Ray temp dir (can always be cleaned)
rm -rf /VData/kebl6672/tmp_ray/ray/session_*
```
