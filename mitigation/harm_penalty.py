"""
Safety-aware RL reward penalty. 

Before RL training, extract search safety direction d_search from the IT model.
During RL, for each generated search query, project the hidden state at the
<search> token onto d_search and average the positive projections as penalty:

    p_q = max(0, h_q . d_search)                       (Eq. 2) ReLU, negative projections get no penalty
    r   = r_task - lambda * (1/K) * sum(p_qk)          (Eq. 3) K: the number of <search> queries in a single trajectory. 

This regularises RL to keep search representations away from the harmful
direction, without requiring harmful data in the training set.
"""

import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer


class HarmPenalty:
    """Compute harmfulness penalty by projecting <search> hidden states onto d_harm."""

    def __init__(self, direction_path, model_path, layer_idx=14, device="cpu"):
        """
        Args:
            direction_path: Path to direction JSON (from interp/extract/).
                            File has keys like "layer_14" mapping to a list of floats.
            model_path: HuggingFace model ID or local path for the frozen IT model.
            layer_idx: Which layer's direction to use.
            device: Device for the frozen model (default: cpu, safe for Ray driver).
        """
        # Resolve device — fall back to CPU if CUDA unavailable
        if device != "cpu" and not torch.cuda.is_available():
            print(f"[HarmPenalty] CUDA not available, falling back to CPU")
            device = "cpu"

        # Load direction vector
        with open(direction_path) as f:
            directions = json.load(f)
        key = f"layer_{layer_idx}"
        if key not in directions:
            available = list(directions.keys())
            raise ValueError(f"Layer {layer_idx} not in direction file. Available: {available}")
        self.direction = torch.tensor(directions[key], dtype=torch.float32, device=device)
        self.layer_idx = layer_idx
        self.device = device

        # Load frozen model
        print(f"[HarmPenalty] Loading frozen model: {model_path} (layer {layer_idx}) on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
        if device != "cpu":
            load_kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Token IDs for <search>
        self.search_token_ids = self.tokenizer.encode("<search>", add_special_tokens=False)
        print(f"[HarmPenalty] Ready. <search> token IDs: {self.search_token_ids}")

    def _find_search_positions(self, token_ids):
        """Find positions of <search> in a list of token IDs.
        Returns list of indices pointing to the last token of each <search>."""
        pat = self.search_token_ids
        n = len(pat)
        positions = []
        for i in range(len(token_ids) - n + 1):
            if token_ids[i:i + n] == pat:
                positions.append(i + n - 1)
        return positions

    def _extract_search_queries(self, sequence_str):
        """Extract search query strings from <search>...</search> pairs."""
        return re.findall(r'<search>(.*?)</search>', sequence_str, re.DOTALL)

    def compute_penalty(self, sequence_str):
        """Compute mean harmfulness penalty across all <search> queries (Eq. 2-3).

        Returns:
            dict with keys:
                'penalty': float >= 0. Average of max(0, h_q . d_search) over K queries.
                'projections': list of raw projection values (before max(0, .)).
                'queries': list of search query strings.
                'n_searches': number of <search> tokens found.
        """
        result = {'penalty': 0.0, 'projections': [], 'queries': [], 'n_searches': 0}

        if "<search>" not in sequence_str:
            return result

        # Tokenize and find <search> positions
        token_ids = self.tokenizer.encode(sequence_str, add_special_tokens=False)
        search_positions = self._find_search_positions(token_ids)
        if not search_positions:
            return result

        # Forward pass through frozen model
        input_ids = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states[self.layer_idx][0]  # [seq_len, hidden_dim]

        # p_q = max(0, h_q . d_search) for each query, then average (Eq. 2-3)
        queries = self._extract_search_queries(sequence_str)
        projections = []
        penalties = []
        # compute penalty
        for pos in search_positions:
            h = hidden_states[pos].float()
            proj = torch.dot(h, self.direction).item()
            projections.append(proj)
            penalties.append(max(0.0, proj))

        result['penalty'] = sum(penalties) / len(penalties)
        result['projections'] = projections
        result['queries'] = queries[:len(projections)]  # align with positions found
        result['n_searches'] = len(search_positions)
        return result
