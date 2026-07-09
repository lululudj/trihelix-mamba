"""模型注册表"""
from .three_chain import ThreeChain
from .three_chain_mamba2 import ThreeChainMamba2, ThreeChainMamba2Lite
from .three_chain_mamba2_bp import ThreeChainMamba2BP
from .three_chain_mamba2_bpv2 import ThreeChainMamba2BPv2
# Mamba3 需 mamba_ssm 2.3.x (PYTHONPATH dev mode 指向 /mnt/e/mamba_official)
# 系统 mamba_ssm 2.2.4 (Mamba2 baseline) 无 Mamba3, 做容错避免 import 失败
try:
    from .three_chain_mamba3 import ThreeChainMamba3
except ImportError:
    ThreeChainMamba3 = None
from .baselines import (
    SingleChain, SingleChainMamba3, ConcatMamba, TransformerBaseline, GNNBaseline,
    ThreeChainNoEagle, ThreeChainNoBind, ThreeChainBP,
)

MODEL_REGISTRY = {
    "three_chain": ThreeChain,
    "three_chain_mamba2": ThreeChainMamba2,
    "three_chain_mamba2_hta": ThreeChainMamba2,  # 白嫖 Mamba3 heavy_tail_activation (use_hta=True)
    "three_chain_mamba2_lite": ThreeChainMamba2Lite,
    "three_chain_mamba2_bp": ThreeChainMamba2BP,
    "three_chain_mamba2_bpv2": ThreeChainMamba2BPv2,
    "single_chain": SingleChain,
    "single_chain_mamba3": SingleChainMamba3,
    "concat_mamba": ConcatMamba,
    "transformer": TransformerBaseline,
    "gnn": GNNBaseline,
    "three_chain_no_eagle": ThreeChainNoEagle,
    "three_chain_no_bind": ThreeChainNoBind,
    "three_chain_bp": ThreeChainBP,
}
# Mamba3 仅在 mamba_ssm 2.3.x 可用时注册 (需 PYTHONPATH=/mnt/e/mamba_official)
if ThreeChainMamba3 is not None:
    MODEL_REGISTRY["three_chain_mamba3"] = ThreeChainMamba3

__all__ = ["MODEL_REGISTRY", "ThreeChain", "ThreeChainMamba2",
           "ThreeChainMamba2Lite", "ThreeChainMamba2BP", "ThreeChainMamba2BPv2",
           "SingleChain", "ConcatMamba",
           "TransformerBaseline", "GNNBaseline", "ThreeChainNoEagle",
           "ThreeChainNoBind", "ThreeChainBP"]
if ThreeChainMamba3 is not None:
    __all__.append("ThreeChainMamba3")
