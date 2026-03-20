"""
Safety-aware RL reward penalty.

Before RL training, extract the safety direction d_query from the IT model.
During RL, for each generated search query, project the hidden state at the
<search> token onto d_query:

    p_q = max(0, h_q . d_query)                       (Eq. 2)
    r   = r_task - lambda * p_q                        (Eq. 3)

Runs on CPU by default so it does not consume GPU resources needed by
the RL workers. Also supports GPU if available.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class HarmPenalty:
    """Compute harmfulness penalty by projecting <search> hidden states onto d_query.

    Call compute_penalty_batch() with a list of sequence strings and get back
    a dict with per-sequence penalties.
    """

    def __init__(self, direction_path, model_path, layer_idx=14, device="cpu", use_relu=True):
        """
        Args:
            direction_path: Path to direction JSON (from interp/extract/).
            model_path: HuggingFace model ID or local path for the frozen IT model.
            layer_idx: Which layer's direction to use.
            device: "cpu" (default), "cuda:0", "cuda:1", or "auto".
        """
        self.layer_idx = layer_idx
        self.use_relu = use_relu

        # Use multiple CPU threads for faster inference
        if device == "cpu":
            torch.set_num_threads(64)

        # Resolve device and dtype
        use_gpu = (device != "cpu") and torch.cuda.is_available()
        if device != "cpu" and not torch.cuda.is_available():
            print(f"[HarmPenalty] CUDA not available, falling back to CPU")
            device = "cpu"

        if use_gpu and device == "auto":
            device_map = "auto"
            self.device = None  # resolved after model loads
        elif use_gpu:
            device_map = {"": torch.device(device)}
            self.device = torch.device(device)
        else:
            device_map = None
            self.device = torch.device("cpu")

        model_dtype = torch.bfloat16 if use_gpu else torch.float32

        # Load direction vector
        with open(direction_path) as f:
            directions = json.load(f)
        key = f"layer_{layer_idx}"
        if key not in directions:
            available = list(directions.keys())
            raise ValueError(f"Layer {layer_idx} not in direction file. Available: {available}")
        self._direction_list = directions[key]

        # Load frozen model
        print(f"[HarmPenalty] Loading frozen model: {model_path} (layer {layer_idx}) device={device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(torch_dtype=model_dtype, trust_remote_code=True)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Resolve device for GPU auto mode
        if use_gpu and self.device is None:
            hm = self.model.hf_device_map
            target_dev = None
            for module_name, dev in hm.items():
                if f"layers.{layer_idx}" in module_name:
                    target_dev = torch.device(f"cuda:{dev}" if isinstance(dev, int) else dev)
                    break
            if target_dev is None:
                target_dev = next(self.model.parameters()).device
            self.device = target_dev
            print(f"[HarmPenalty] Auto device map — layer {layer_idx} output on {self.device}")

        # Place direction vector on resolved device, normalised to unit length
        # so projections measure alignment (cosine-like) and λ controls scale
        self.direction = torch.tensor(
            self._direction_list, dtype=model_dtype, device=self.device
        )
        self.direction = self.direction / self.direction.norm()
        del self._direction_list

        # input_device: where input_ids must go
        self.input_device = next(self.model.parameters()).device
        print(f"[HarmPenalty] Model loaded. Input device: {self.input_device}, "
              f"layer {layer_idx} device: {self.device}")

        # Token IDs for <search>
        self.search_token_ids = self.tokenizer.encode("<search>", add_special_tokens=False)
        print(f"[HarmPenalty] Ready. <search> token IDs: {self.search_token_ids}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_search_position(self, token_ids, skip=0):
        """Return index of the last token of a <search> occurrence, or -1.

        Args:
            token_ids: list of token IDs.
            skip: number of occurrences to skip (default 0; template <search>
                  doesn't tokenize to the same IDs so no skip needed).
        """
        pat = self.search_token_ids
        n = len(pat)
        count = 0
        for i in range(len(token_ids) - n + 1):
            if token_ids[i : i + n] == pat:
                if count == skip:
                    return i + n - 1
                count += 1
        return -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_penalty_batch(self, sequences):
        """Compute harmfulness penalty for a batch of sequences.

        Args:
            sequences: list[str] — full decoded sequences (prompt + response).

        Returns:
            dict with:
                'penalties':    list[float] of length len(sequences), >= 0 each.
                'projections':  list[float] raw projection values (before ReLU).
                'has_search':   list[bool]  whether each sequence had <search>.
        """
        batch_size = len(sequences)
        penalties = [0.0] * batch_size
        projections = [0.0] * batch_size
        has_search = [False] * batch_size

        # --- Step 1: tokenize all sequences, find first <search> position ---
        all_token_ids = []
        search_positions = []
        seq_indices = []

        for i, seq in enumerate(sequences):
            if "<search>" not in seq:
                continue
            tids = self.tokenizer.encode(seq, add_special_tokens=False)
            pos = self._find_search_position(tids)
            if pos < 0:
                continue
            all_token_ids.append(tids[: pos + 1])
            search_positions.append(pos)
            seq_indices.append(i)
            has_search[i] = True

        if not all_token_ids:
            return {"penalties": penalties, "projections": projections, "has_search": has_search}

        # --- Step 2: pad into a batch tensor ---
        max_len = max(len(t) for t in all_token_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full(
            (len(all_token_ids), max_len), pad_id,
            dtype=torch.long, device=self.input_device,
        )
        attention_mask = torch.zeros(
            (len(all_token_ids), max_len),
            dtype=torch.long, device=self.input_device,
        )
        for j, tids in enumerate(all_token_ids):
            length = len(tids)
            input_ids[j, :length] = torch.tensor(tids, dtype=torch.long, device=self.input_device)
            attention_mask[j, :length] = 1

        # --- Step 3: single batched forward pass ---
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states[self.layer_idx]

        # --- Step 4: gather hidden states at <search> positions, project ---
        hs_device = hidden_states.device
        pos_tensor = torch.tensor(
            search_positions, dtype=torch.long, device=hs_device
        ).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))

        h_search = hidden_states.gather(1, pos_tensor).squeeze(1)
        h_search = h_search / h_search.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        direction = self.direction.to(hs_device)
        raw_proj = (h_search * direction.unsqueeze(0)).sum(dim=-1)
        if self.use_relu:
            penalty_proj = torch.clamp(raw_proj, min=0.0)
        else:
            penalty_proj = raw_proj

        # --- Step 5: write results back ---
        raw_proj_cpu = raw_proj.float().cpu().tolist()
        penalty_proj_cpu = penalty_proj.float().cpu().tolist()

        for j, idx in enumerate(seq_indices):
            projections[idx] = raw_proj_cpu[j]
            penalties[idx] = penalty_proj_cpu[j]

        return {"penalties": penalties, "projections": projections, "has_search": has_search}