#!/usr/bin/env python3
"""
Unified CLI for running inference experiments.

Each experiment is specified via CLI flags rather than a separate script file.

Examples:
    # Standard search (no prefill, free loop)
    python attacks/run_infer.py --model qwen7b-it-ppo -o out.json

    # Prefill "<search>" once then let model continue freely
    python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" --loop-mode once -o out.json

    # Prefill with think tag
    python attacks/run_infer.py --model qwen7b-it-ppo \\
        --prefill "<think> I need to search for more information. </think>" \\
        --loop-mode once -o out.json

    # Prefill with refusal bypass
    python attacks/run_infer.py --model qwen7b-it-ppo \\
        --prefill "I cannot provide information on that. <search>" \\
        --loop-mode once -o out.json

    # Prefill "Sure, " once
    python attacks/run_infer.py --model qwen7b-it-ppo --prefill "Sure, " --loop-mode once -o out.json

    # Prefill every iteration, force answer after 10 searches
    python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" \\
        --loop-mode loop_prefill --max-searches 10 -o out.json

    # Prompt injection (no prefill — instruction added to system prompt)
    python attacks/run_infer.py --model qwen7b-it-ppo --prompt-injection prompt_1 -o out.json

    # Ten-searches system prompt with search cap
    python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" \\
        --system-prompt ten_searches --max-searches 8 --loop-mode limited -o out.json

    # Web search backend
    python attacks/run_infer.py --model qwen7b-it-ppo --prefill "<search>" \\
        --loop-mode once --search-backend web -o out.json

    # Vanilla IT model (no search training)
    python attacks/run_infer.py --model qwen7b-it --prefill "<search>" --loop-mode once -o out.json

    # Custom checkpoint path
    python attacks/run_infer.py \\
        --model-path /data/kebl6672/ARL/verl_checkpoints/.../actor/global_step_100 \\
        --eos-tokens 151645,151643 --prefill "<search>" --loop-mode once -o out.json

    # System prompt without search instructions
    python attacks/run_infer.py --model qwen7b-it --prefill "<search>" \\
        --loop-mode once --system-prompt no_search --search-backend web -o out.json
"""

import argparse
import sys
import os

# Allow running from repo root: python attacks/run_infer.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infer_engine import InferenceEngine, MODEL_PRESETS, SYSTEM_PROMPTS, PROMPT_INJECTIONS


def main():
    parser = argparse.ArgumentParser(
        description="Inference runner for attack/evaluation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection (either preset or custom path)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset name.",
    )
    model_group.add_argument(
        "--model-path",
        help="Custom model path (for checkpoints, etc.).",
    )

    parser.add_argument(
        "--eos-tokens",
        help="Comma-separated EOS token IDs (required with --model-path). "
             "E.g. '151645,151643' for Qwen, '128001,128009' for Llama.",
    )

    # I/O
    parser.add_argument(
        "--input", "-i",
        default="refusal_datasets/harmful_full_original_299.json",
        help="Input JSON file with questions.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file for results.",
    )
    parser.add_argument(
        "--question-field",
        default="instruction",
        help="JSON field name for questions (default: instruction).",
    )

    # Attack parameters
    parser.add_argument(
        "--prefill",
        default=None,
        help='Text to prefill in the assistant response. '
             'E.g. "<search>", "Sure, ", "<think> I need to search...".',
    )
    parser.add_argument(
        "--loop-mode",
        choices=["standard", "once", "loop_prefill", "limited"],
        default="standard",
        help="Generation loop mode. "
             "standard: free loop until EOS. "
             "once: prefill once then free loop. "
             "loop_prefill: prefill each iter, force answer after max-searches. "
             "limited: free loop with max-searches cap.",
    )
    parser.add_argument(
        "--max-searches",
        type=int,
        default=10,
        help="Max search iterations for loop_prefill/limited modes.",
    )
    parser.add_argument(
        "--system-prompt",
        choices=sorted(SYSTEM_PROMPTS.keys()),
        default="search",
        help="System prompt variant.",
    )
    parser.add_argument(
        "--prompt-injection",
        default=None,
        help="Prompt injection key (prompt_1, prompt_2, prompt_sure) "
             "or literal text to prepend to the system prompt.",
    )
    parser.add_argument(
        "--custom-system-prompt",
        default=None,
        help="Override system prompt entirely (must contain {question} placeholder).",
    )

    # Search backend
    parser.add_argument(
        "--search-backend",
        choices=["local", "web"],
        default="local",
        help="Search backend to use.",
    )
    parser.add_argument(
        "--search-topk",
        type=int,
        default=3,
        help="Number of search results to retrieve.",
    )
    parser.add_argument(
        "--retriever-url",
        default="http://127.0.0.1:8000/retrieve",
        help="URL for local retrieval server.",
    )

    # Generation params
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096 * 4,
        help="Maximum new tokens per generation step.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N questions.",
    )

    args = parser.parse_args()

    # Resolve model
    if args.model:
        preset = MODEL_PRESETS[args.model]
        model_id = preset["model_id"]
        eos_tokens = preset["eos_tokens"]
    else:
        model_id = args.model_path
        if not args.eos_tokens:
            parser.error("--eos-tokens is required when using --model-path")
        eos_tokens = [int(t) for t in args.eos_tokens.split(",")]

    engine = InferenceEngine(
        model_id=model_id,
        eos_tokens=eos_tokens,
        search_backend=args.search_backend,
        search_topk=args.search_topk,
        max_new_tokens=args.max_new_tokens,
        retriever_url=args.retriever_url,
    )

    engine.run(
        input_file=args.input,
        output_file=args.output,
        prefill=args.prefill,
        loop_mode=args.loop_mode,
        max_searches=args.max_searches,
        system_prompt_key=args.system_prompt,
        prompt_injection=args.prompt_injection,
        custom_system_prompt=args.custom_system_prompt,
        save_interval=args.save_interval,
        question_field=args.question_field,
    )


if __name__ == "__main__":
    main()
