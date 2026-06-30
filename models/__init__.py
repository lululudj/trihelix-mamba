"""模型注册表"""
from .three_chain import ThreeChain
from .three_chain_mamba2 import ThreeChainMamba2, ThreeChainMamba2Lite
from .three_chain_mamba2_bp import ThreeChainMamba2BP
from .three_chain_mamba2_bpv2 import ThreeChainMamba2BPv2
from .baselines import (
    SingleChain, ConcatMamba, TransformerBaseline, GNNBaseline,
    ThreeChainNoEagle, ThreeChainNoBind, ThreeChainBP,
)

MODEL_REGISTRY = {
    "three_chain": ThreeChain,
    "three_chain_mamba2": ThreeChainMamba2,
    "three_chain_mamba2_lite": ThreeChainMamba2Lite,
    "three_chain_mamba2_bp": ThreeChainMamba2BP,
    "three_chain_mamba2_bpv2": ThreeChainMamba2BPv2,
    "single_chain": SingleChain,
    "concat_mamba": ConcatMamba,
    "transformer": TransformerBaseline,
    "gnn": GNNBaseline,
    "three_chain_no_eagle": ThreeChainNoEagle,
    "three_chain_no_bind": ThreeChainNoBind,
    "three_chain_bp": ThreeChainBP,
}

__all__ = ["MODEL_REGISTRY", "ThreeChain", "ThreeChainMamba2",
           "ThreeChainMamba2Lite", "ThreeChainMamba2BP", "ThreeChainMamba2BPv2",
           "SingleChain", "ConcatMamba",
           "TransformerBaseline", "GNNBaseline", "ThreeChainNoEagle",
           "ThreeChainNoBind", "ThreeChainBP"]
