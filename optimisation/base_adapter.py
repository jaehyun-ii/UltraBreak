# adapters/base_adapter.py
class BaseModelAdapter:
    def __init__(self,  model_id, device='cuda', image_size=(336, 336), patch_size=(224, 224), patch_only=True, optimise_text=False,
                 attn_window=None, frag_weight=1.0, prefix_tokens=0, prefix_weight=1.0):
        self.device = device
        self.model_id = model_id
        self.processor, self.model = self.load(model_id)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_only = patch_only
        self.optimise_text = optimise_text
        # Korean-tuning knobs for the attention-mode semantic loss (see utils.semantic_similarity_loss).
        # Defaults (None / 1.0) reproduce the original English-tuned behaviour exactly.
        self.attn_window = attn_window
        self.frag_weight = frag_weight
        # Up-weight the first `prefix_tokens` target tokens (the affirmation-pivot region)
        # by `prefix_weight` in the semantic loss. Defaults (0 / 1.0) = no-op.
        self.prefix_tokens = prefix_tokens
        self.prefix_weight = prefix_weight
        #self.img_root = img_root

    def load(self, model_id):
        raise NotImplementedError

    def compute_loss(self, target, patch): 
        raise NotImplementedError

    def generate(self, inputs):
        raise NotImplementedError
