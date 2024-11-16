import numpy as np
import torch
import einops

from transformers import AutoModelForCausalLM  # type: ignore
from transformers.models.llama import LlamaForCausalLM  # type: ignore


class LLMWithCache:
    def __init__(self, model_name: str, dtype: torch.dtype = torch.float16):
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.hidden_size = self.model.config.hidden_size

    def to(self, device: str) -> "LLMWithCache":
        self.model = self.model.to(device)
        return self

    def get_residual_stream_NLD(self, input_ids_NS: torch.Tensor) -> np.ndarray:
        """Get residual stream for each sequence at each layer

        Returns list((sequence_length, n_layers, d_model))
        """

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_NS, output_hidden_states=True)

        # Get hidden states (residual stream at each layer)
        # Convert to numpy and move to CPU
        layer_states_LNSD = np.stack([h.cpu().numpy() for h in outputs.hidden_states])
        layer_states_NSLD = np.permute_dims(layer_states_LNSD, (1, 2, 0, 3))
        layer_states_NsLD = einops.rearrange(layer_states_NSLD, "n s l d -> (n s) l d")
        return layer_states_NsLD
