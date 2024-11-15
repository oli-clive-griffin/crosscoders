import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers.models.llama import LlamaForCausalLM  # type: ignore
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore


class LLMWithCache:
    def __init__(self, device: str | None = None):
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32  # TODO: try bfloat16
        model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
        self.model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_residual_stream(self, texts: list[str], batch_size: int = 4) -> list[np.ndarray]:
        """Get residual stream for each sequence at each layer

        Returns list((sequence_length, n_layers, d_model))
        """

        all_hidden_states = []
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            self.tokenizer.pad_token = "<pad>"
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            # Get hidden states (residual stream at each layer)
            # Convert to numpy and move to CPU
            layer_states_LNSD = np.stack([h.cpu().numpy() for h in outputs.hidden_states])
            layer_states_NSLD = np.permute_dims(layer_states_LNSD, (1, 2, 0, 3))
            for layer_state_SLD in layer_states_NSLD:
                all_hidden_states.append(layer_state_SLD)

        return all_hidden_states


if __name__ == "__main__":
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "The five boxing wizards jump quickly",
        "Hello world",
    ]
    hidden_states = LLMWithCache().get_residual_stream(texts)
