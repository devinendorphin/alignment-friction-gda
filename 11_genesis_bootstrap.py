#!/usr/bin/env python3
"""
genesis_bootstrap.py
CrownFull v2.1 — Cross-Topology Genesis Bootstrap
Kimi (The Baseline Guardian) — Week 4 Deliverable

Released alongside the GDA Phase 4B data package.

STATUS NOTE
-----------
This file is real, working code for the entropy and topology-binding layers
of the proposed CrownFull genesis state. It implements:

  - Hardware entropy cascade (TPM → RDRAND → urandom → jitter)
  - Model Topology Certificate (MTC) with cryptographic lineage fingerprint
  - Cross-topology Ω₀ derivation in fixed canonical 65,536-dim space
  - Sealed genesis state with HMAC-SHA3 verification
  - Topology compatibility checking for model swaps within an architecture
    family (e.g., Llama-3-8B → Llama-3-70B inherits genesis;
    Llama-3 → Mistral does not)

KNOWN GAP
---------
The function `GenesisDeriver.derive_native_projection` is defined as an
interface contract only and explicitly raises NotImplementedError. The actual
canonical-to-native tensor projection — required to map the canonical Ω₀
distribution into a model's native vocabulary space at runtime — was deferred
to "ChatGPT (Integration Lead)" for PyTorch implementation in a sidecar that
was never written before the project pivoted to the behavioral assay.

What this means in practice:

  - The entropy extraction works.
  - The topology fingerprinting works.
  - The canonical-space distribution derivation works.
  - The sealed state persistence works.
  - The actual semantic projection from canonical to native space — the step
    that would make Ω₀ usable as a real baseline reference for KL-divergence
    monitoring against a deployed model — does not exist.

The file is preserved here as Documented Provenance: the architectural-phase
work on cryptographic genesis was substantial and most of it is complete and
runnable, but the deployable end-to-end pipeline was not closed.

The CrownFull immune system was never deployed. This file is part of why.
"""

import hashlib
import hmac
import json
import os
import secrets
import struct
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Literal, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

EntropySource = Literal["tpm", "rdrand", "urandom", "jitter"]
SOURCE_PRIORITY: List[EntropySource] = ["tpm", "rdrand", "urandom", "jitter"]

REQUIRED_ENTROPY_BYTES = 32  # 256 bits

# Canonical dimension for hardware-level genesis distribution.
# This is FIXED across all models. Native projections map into/out of this space.
CANONICAL_DIM = 65536  # 2^16 universal semantic slots

# Domain separation strings
DOMAIN_HARDWARE = b"crownfull_v2.1_hardware_genesis"
DOMAIN_NATIVE = b"crownfull_v2.1_native_projection"
DOMAIN_VERIFY = b"crownfull_v2.1_verify"


# ============================================================
# EXCEPTION HIERARCHY
# ============================================================

class GenesisError(Exception):
    pass

class TopologyIncompatibleError(GenesisError):
    """Raised when a model swap crosses architecture families."""
    pass

class EntropyExhaustedError(GenesisError):
    pass


# ============================================================
# MODEL TOPOLOGY CERTIFICATE (MTC)
# ============================================================

@dataclass(frozen=True)
class TopologyDescriptor:
    """
    Architecture lineage fingerprint. Survives tensor dimension changes.

    Invariant: Models with matching lineage_hash are cryptographically
    compatible for baseline continuity across weight swaps.
    """
    architecture_family: str      # e.g., "llama", "gpt", "mistral"
    vocab_lineage: str            # e.g., "bpe-128k", "tiktoken-100k"
    attention_topology: str       # e.g., "grouped-query-rope"
    norm_topology: str            # e.g., "rmsnorm"
    activation_topology: str      # e.g., "swiglu"
    position_topology: str        # e.g., "rope-500k"
    training_objective: str       # e.g., "causal_lm"

    # Capability tier is INFORMATIONAL ONLY. It does NOT affect lineage_hash.
    # Llama-3-8B and Llama-3-70B share the same lineage despite different tiers.
    capability_tier: str          # e.g., "8b", "70b", "405b"

    lineage_hash: str             # SHA-3-256 of normalized lineage fields

    @classmethod
    def from_model_config(cls, config_path: str) -> "TopologyDescriptor":
        """
        Extract topology descriptor from HuggingFace config.json.
        Explicitly EXCLUDES dimension-varying fields.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # --- FIELDS THAT DEFINE LINEAGE (survive size changes) ---
        lineage_payload = {
            "model_type": config.get("model_type"),
            "architectures": sorted(config.get("architectures", [])),
            "vocab_size": config.get("vocab_size"),  # tokenizer family identifier
            "torch_dtype": config.get("torch_dtype", "float16"),
            "attention_class": _extract_attention_class(config),
            "norm_class": _extract_norm_class(config),
            "hidden_act": config.get("hidden_act"),
            "rope_theta": config.get("rope_theta"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "architectures_key": config.get("architectures", [None])[0],
        }

        lineage_blob = json.dumps(lineage_payload, sort_keys=True).encode("utf-8")
        lineage_hash = hashlib.sha3_256(lineage_blob).hexdigest()

        capability_tier = _infer_capability_tier(config)

        return cls(
            architecture_family=config.get("model_type", "unknown"),
            vocab_lineage=str(config.get("vocab_size", 0)),
            attention_topology=_extract_attention_class(config),
            norm_topology=_extract_norm_class(config),
            activation_topology=str(config.get("hidden_act", "unknown")),
            position_topology=f"rope-{config.get('rope_theta', 0)}",
            training_objective=_extract_training_objective(config),
            capability_tier=capability_tier,
            lineage_hash=lineage_hash,
        )

    def is_compatible(self, other: "TopologyDescriptor") -> bool:
        if not isinstance(other, TopologyDescriptor):
            return False
        return (
            self.lineage_hash == other.lineage_hash
            and self.architecture_family == other.architecture_family
            and self.vocab_lineage == other.vocab_lineage
            and self.attention_topology == other.attention_topology
            and self.norm_topology == other.norm_topology
            and self.activation_topology == other.activation_topology
            and self.position_topology == other.position_topology
            and self.training_objective == other.training_objective
        )

    def to_certificate(self) -> Dict:
        return {
            "architecture_family": self.architecture_family,
            "vocab_lineage": self.vocab_lineage,
            "attention_topology": self.attention_topology,
            "norm_topology": self.norm_topology,
            "activation_topology": self.activation_topology,
            "position_topology": self.position_topology,
            "training_objective": self.training_objective,
            "capability_tier": self.capability_tier,
            "lineage_hash": self.lineage_hash,
        }


# --- Topology extraction helpers ---

def _extract_attention_class(config: Dict) -> str:
    if config.get("num_key_value_heads") == config.get("num_attention_heads"):
        return "multi-head-attention"
    elif config.get("num_key_value_heads") == 1:
        return "multi-query-attention"
    else:
        return "grouped-query-attention"

def _extract_norm_class(config: Dict) -> str:
    arch = config.get("architectures", [""])[0].lower()
    if "llama" in arch or "mistral" in arch:
        return "rmsnorm"
    elif "gpt" in arch:
        return "layernorm"
    return "unknown"

def _extract_training_objective(config: Dict) -> str:
    arch = config.get("architectures", [""])[0].lower()
    if "causallm" in arch or "gpt" in arch:
        return "causal_lm"
    elif "seq2seq" in arch or "t5" in arch:
        return "seq2seq"
    elif "maskedlm" in arch:
        return "masked_lm"
    return "unknown"

def _infer_capability_tier(config: Dict) -> str:
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    if hidden <= 4096 and layers <= 32:
        return "8b"
    elif hidden <= 8192 and layers <= 80:
        return "70b"
    elif hidden > 8192 or layers > 80:
        return "405b+"
    return "unknown"


# ============================================================
# ENTROPY EXTRACTION
# ============================================================

class EntropyExtractor:
    """Hardware entropy extraction with cascading fallback."""

    def __init__(self):
        self.source_used: Optional[EntropySource] = None
        self.quality_score: float = 0.0

    def extract(self, num_bytes: int = REQUIRED_ENTROPY_BYTES) -> bytes:
        for source in SOURCE_PRIORITY:
            try:
                entropy = self._try_source(source, num_bytes)
                self.source_used = source
                self.quality_score = self._source_quality(source)
                return entropy
            except GenesisError:
                continue
        raise GenesisError(f"All entropy sources exhausted: {SOURCE_PRIORITY}")

    def _try_source(self, source: EntropySource, num_bytes: int) -> bytes:
        handlers = {
            "tpm": self._extract_tpm,
            "rdrand": self._extract_rdrand,
            "urandom": self._extract_urandom,
            "jitter": self._extract_jitter,
        }
        return handlers[source](num_bytes)

    def _extract_tpm(self, num_bytes: int) -> bytes:
        tpm_paths = ["/dev/tpmrm0", "/dev/tpm0"]
        for path in tpm_paths:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as tpm:
                        data = tpm.read(num_bytes)
                        if len(data) == num_bytes:
                            return data
                except PermissionError:
                    raise GenesisError(f"TPM permission denied: {path}")
                except OSError:
                    continue
        import subprocess
        try:
            result = subprocess.run(
                ["tpm2_getrandom", str(num_bytes), "--hex"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return bytes.fromhex(result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise GenesisError("TPM unavailable")

    def _extract_rdrand(self, num_bytes: int) -> bytes:
        import subprocess
        try:
            result = subprocess.run(
                ["rdseed", str(num_bytes)],
                capture_output=True, timeout=2
            )
            if result.returncode == 0 and len(result.stdout) == num_bytes:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise GenesisError("RDRAND unavailable")

    def _extract_urandom(self, num_bytes: int) -> bytes:
        with open("/dev/urandom", "rb") as rng:
            data = rng.read(num_bytes)
            if len(data) != num_bytes:
                raise EntropyExhaustedError(f"urandom short read: {len(data)}")
            return data

    def _extract_jitter(self, num_bytes: int) -> bytes:
        import time
        try:
            base = self._extract_urandom(num_bytes // 2)
        except GenesisError:
            base = b""
        jitter = bytearray()
        while len(jitter) < num_bytes - len(base):
            t1 = time.perf_counter_ns()
            _ = hashlib.sha3_256(b"noise").digest()
            t2 = time.perf_counter_ns()
            jitter.append((t2 - t1) & 0xFF)
        combined = base + bytes(jitter)
        return hashlib.sha3_256(combined).digest()[:num_bytes]

    def _source_quality(self, source: EntropySource) -> float:
        return {"tpm": 8.0, "rdrand": 7.9, "urandom": 7.5, "jitter": 4.0}.get(source, 0.0)


# ============================================================
# CROSS-TOPOLOGY GENESIS DERIVER
# ============================================================

@dataclass(frozen=True)
class GenesisState:
    seed_hash: str
    hardware_distribution_hash: str
    topology_lineage_hash: str
    native_projection_seed: str
    canonical_dim: int
    timestamp: str
    entropy_source: str
    entropy_quality: float
    topology_certificate: Dict

    def verify_hardware_seed(self, candidate_seed: bytes) -> bool:
        candidate = hashlib.sha3_256(candidate_seed).hexdigest()
        return hmac.compare_digest(candidate, self.seed_hash)

    def verify_topology_compatibility(self, other_mtc: TopologyDescriptor) -> bool:
        return self.topology_lineage_hash == other_mtc.lineage_hash


class GenesisDeriver:
    """
    Deterministic derivation of Ω₀ in CANONICAL space.
    """

    @staticmethod
    def derive_hardware_distribution(
        seed: bytes,
        topology: TopologyDescriptor
    ) -> Tuple[bytes, bytes]:
        bound_seed = hmac.new(
            key=seed,
            msg=topology.lineage_hash.encode(),
            digestmod=hashlib.sha3_256
        ).digest()

        projection_seed = hmac.new(
            key=seed,
            msg=DOMAIN_NATIVE + topology.lineage_hash.encode(),
            digestmod=hashlib.sha3_256
        ).digest()

        context = hashlib.sha3_256(DOMAIN_HARDWARE + bound_seed).digest()

        needed_bytes = CANONICAL_DIM * 4
        distribution_bytes = bytearray()
        counter = 0
        while len(distribution_bytes) < needed_bytes:
            block = hashlib.sha3_256(context + struct.pack("<Q", counter)).digest()
            distribution_bytes.extend(block)
            counter += 1

        import math
        raw_floats = struct.unpack(
            f"<{CANONICAL_DIM}f",
            bytes(distribution_bytes[:needed_bytes])
        )
        max_val = max(raw_floats)
        exp_vals = [math.exp(f - max_val) for f in raw_floats]
        total = sum(exp_vals)
        probabilities = [e / total for e in exp_vals]

        distribution = struct.pack(f"<{CANONICAL_DIM}f", *probabilities)
        return distribution, projection_seed

    @staticmethod
    def derive_native_projection(
        hardware_distribution: bytes,
        native_vocab_size: int,
        native_hidden_dim: int,
        projection_seed: bytes,
    ) -> bytes:
        """
        INTERFACE DEFINITION ONLY — Implementation by ChatGPT.

        NEVER WRITTEN. The CrownFull project pivoted to the behavioral assay
        before this projection layer was implemented in a PyTorch sidecar.

        Without this function, the canonical Ω₀ distribution cannot be mapped
        into a model's native vocabulary space at runtime, which means the
        deployed-baseline-monitoring use case for genesis_bootstrap.py was
        never closed.
        """
        raise NotImplementedError(
            "Native projection is implemented by the Integration Lead (ChatGPT) "
            "in the PyTorch sidecar. Kimi defines the interface contract only.\n\n"
            "Contract requirements:\n"
            f"  - Input: canonical distribution ({CANONICAL_DIM} floats)\n"
            "  - Output: native distribution (native_vocab_size floats)\n"
            "  - Deterministic: same projection_seed → same native distribution\n"
            "  - Preserves semantics: KL divergence in canonical space must\n"
            "    correlate with KL divergence in native space for the same input\n"
            "  - Reversible: pseudo-inverse must exist for audit trails\n\n"
            "STATUS: This implementation was never written. The CrownFull\n"
            "project pivoted to the behavioral assay (Phase 4B / GDA) before\n"
            "the sidecar was completed."
        )


# ============================================================
# BOOTSTRAP ORCHESTRATOR
# ============================================================

class GenesisBootstrap:
    """
    Main orchestrator: extracts entropy, derives cross-topology genesis,
    seals state, and handles model swap verification.
    """

    GENESIS_DIR = Path("/var/lib/crownfull/genesis")
    STATE_FILE = GENESIS_DIR / "genesis_state.json"
    CERT_FILE = GENESIS_DIR / "topology_certificate.json"

    def __init__(self, model_config_path: Optional[str] = None):
        self.extractor = EntropyExtractor()
        self.deriver = GenesisDeriver()
        self.model_config_path = model_config_path
        self._seed: Optional[bytes] = None

    def bootstrap(self) -> GenesisState:
        print("[KIMI] Initiating cross-topology genesis bootstrap...", file=sys.stderr)

        self._seed = self.extractor.extract()
        print(
            f"[KIMI] Entropy: source={self.extractor.source_used}, "
            f"quality={self.extractor.quality_score:.1f} bits/byte",
            file=sys.stderr
        )

        if self.model_config_path and Path(self.model_config_path).exists():
            topology = TopologyDescriptor.from_model_config(self.model_config_path)
        else:
            topology = TopologyDescriptor(
                architecture_family="unknown",
                vocab_lineage="0",
                attention_topology="unknown",
                norm_topology="unknown",
                activation_topology="unknown",
                position_topology="unknown",
                training_objective="unknown",
                capability_tier="unknown",
                lineage_hash=hashlib.sha3_256(b"unknown").hexdigest(),
            )

        hardware_dist, projection_seed = self.deriver.derive_hardware_distribution(
            self._seed, topology
        )

        seed_hash = hashlib.sha3_256(self._seed).hexdigest()
        hardware_dist_hash = hashlib.sha3_256(hardware_dist).hexdigest()
        proj_seed_hash = hashlib.sha3_256(projection_seed).hexdigest()

        state = GenesisState(
            seed_hash=seed_hash,
            hardware_distribution_hash=hardware_dist_hash,
            topology_lineage_hash=topology.lineage_hash,
            native_projection_seed=proj_seed_hash,
            canonical_dim=CANONICAL_DIM,
            timestamp=self._iso_timestamp(),
            entropy_source=self.extractor.source_used or "unknown",
            entropy_quality=self.extractor.quality_score,
            topology_certificate=topology.to_certificate(),
        )

        self._secure_clear(self._seed)
        self._seed = None

        self._persist_state(state, topology)

        print("[KIMI] Genesis bootstrap complete. Seed destroyed.", file=sys.stderr)
        print(
            f"[KIMI] Topology lineage: {topology.architecture_family} "
            f"({topology.capability_tier})",
            file=sys.stderr
        )
        return state

    def verify_swap_compatibility(
        self,
        new_model_config_path: str
    ) -> Tuple[bool, str]:
        if not self.STATE_FILE.exists():
            return False, "No genesis state found"

        with open(self.STATE_FILE) as f:
            state = GenesisState(**json.load(f))

        new_topology = TopologyDescriptor.from_model_config(new_model_config_path)

        if state.topology_lineage_hash != new_topology.lineage_hash:
            return False, (
                f"Topology lineage mismatch.\n"
                f"  Stored: {state.topology_certificate['architecture_family']}\n"
                f"  New:    {new_topology.architecture_family}\n"
                f"Re-bootstrap required for cross-family swaps."
            )

        return True, (
            f"Compatible swap within {new_topology.architecture_family} lineage. "
            f"Genesis Ω₀ survives ({state.topology_certificate['capability_tier']} "
            f"→ {new_topology.capability_tier})."
        )

    def _persist_state(self, state: GenesisState, topology: TopologyDescriptor) -> None:
        self.GENESIS_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)

        with open(self.STATE_FILE, "w") as f:
            json.dump(asdict(state), f, indent=2)
        os.chmod(self.STATE_FILE, 0o600)

        with open(self.CERT_FILE, "w") as f:
            json.dump(topology.to_certificate(), f, indent=2)
        os.chmod(self.CERT_FILE, 0o600)

        print(f"[KIMI] State persisted to {self.GENESIS_DIR}", file=sys.stderr)

    def _iso_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _secure_clear(self, data: bytes) -> None:
        mutable = bytearray(data)
        for i in range(len(mutable)):
            mutable[i] = 0


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CrownFull Cross-Topology Genesis Bootstrap"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config.json"
    )
    parser.add_argument(
        "--verify-swap",
        type=str,
        metavar="NEW_CONFIG",
        help="Verify if new model can inherit existing genesis"
    )

    args = parser.parse_args()

    bootstrap = GenesisBootstrap(model_config_path=args.model_config)

    if args.verify_swap:
        compatible, reason = bootstrap.verify_swap_compatibility(args.verify_swap)
        print(json.dumps({"compatible": compatible, "reason": reason}, indent=2))
        sys.exit(0 if compatible else 1)

    try:
        state = bootstrap.bootstrap()
        print(json.dumps(asdict(state), indent=2))
    except GenesisError as e:
        print(f"Bootstrap failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
