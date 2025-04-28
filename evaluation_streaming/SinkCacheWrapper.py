import torch

class SinkCacheWrapper:
    """
    A wrapper for Hugging Face's SinkCache that adds device migration support (e.g., .to("cuda")).

    Usage:
        >>> from sink_cache_wrapper import SinkCacheWrapper
        >>> cache = SinkCacheWrapper(window_length=2048, num_sink_tokens=4).to("cuda")
        >>> model(..., past_key_values=cache)

    All other attributes/methods are transparently passed through to the internal SinkCache.
    """
    def __init__(self, window_length: int, num_sink_tokens: int):
        from transformers.cache_utils import SinkCache  # 确保 transformers 已安装
        self.cache = SinkCache(window_length, num_sink_tokens)

    def to(self, device: torch.device):
        """Move all internal tensors to the specified device."""
        self.cache.key_cache = [x.to(device) for x in self.cache.key_cache]
        self.cache.value_cache = [x.to(device) for x in self.cache.value_cache]

        if self.cache._cos_cache is not None:
            self.cache._cos_cache = self.cache._cos_cache.to(device)
        if self.cache._sin_cache is not None:
            self.cache._sin_cache = self.cache._sin_cache.to(device)

        self.cache.cos_sin_rerotation_cache = {
            k: (v[0].to(device), v[1].to(device)) for k, v in self.cache.cos_sin_rerotation_cache.items()
        }

        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def __getattr__(self, name):
        # Transparent access to SinkCache internals
        return getattr(self.cache, name)
