try:
    from transformers.processing_utils import MultiModalData
except ImportError:
    import transformers.processing_utils
    class MultiModalData:
        pass
    transformers.processing_utils.MultiModalData = MultiModalData

try:
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
except Exception:
    pass
