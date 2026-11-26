# IPM Utility Functions
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

try:
    from qlsas import ResourceEstimator, amplitude_encoding, draw, draw_encoding
    from qlsas.ft import PhysicalResourceEstimator, preset
    QLSAS_AVAILABLE = True
except ImportError:
    ResourceEstimator = None
    amplitude_encoding = None
    draw = None
    draw_encoding = None
    PhysicalResourceEstimator = None
    preset = None
    QLSAS_AVAILABLE = False


def pad_matrix(matrix: np.ndarray, size: int | None = None) -> np.ndarray:
    """Embed a square matrix in a power-of-two size identity matrix."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"matrix must be nonempty and square, got shape {matrix.shape}.")

    n = matrix.shape[0]
    if size is None:
        size = 1 << (n - 1).bit_length()
    if size < n or size & (size - 1):
        raise ValueError(f"size must be a power of two at least {n}, got {size}.")

    padded = np.eye(size, dtype=np.result_type(matrix.dtype, float))
    padded[:n, :n] = matrix
    return padded


def print_qlsa_stats(alg, enc, prob):
    """Print logical QLSA estimates and one-application encoding resources."""
    est = ResourceEstimator().estimate(alg, enc, prob.b, amplitude_encoding)
    print(f"Circuit depth: {est.circuit_depth}")
    print(f"Logical qubits: {est.logical_qubits}")
    print(f"Gate counts: {est.gate_counts}")
    print(f"Oracle calls: {est.oracle_calls}")
    print(f"Success probability: {est.success_probability}")
    print(f"Calls per success: {est.calls_per_success}")
    print(f"Kappa: {est.kappa}")
    print(f"Encoded kappa: {est.kappa_enc}")
    print(f"Subnormalization: {est.subnormalization}")

    enc_resources = enc.resources()
    print("Matrix block encoding (one application):")
    print(f"Qubits (peak allocated): {enc_resources.qubits}")
    print(f"Depth: {enc_resources.depth}")
    print(f"Gate counts: {enc_resources.gate_counts}")

def draw_qlsa(alg, enc, prob, iteration):
    """Save logical, gate-level, and encoding circuit diagrams for an iteration."""
    os.makedirs("drawings", exist_ok=True)
    draw(alg, enc, prob.b, steps=2
         ).to_qiskit().draw("mpl", fold=40, filename=f"drawings/qlsa-logical-{iteration}.png")

    draw(alg, enc, prob.b, level="ladders", steps=1
         ).to_qiskit().draw("mpl", fold=40, filename=f"drawings/qlsa-ladders-{iteration}.png")

    draw(alg, enc, prob.b, level="gates", steps=1
         ).to_qiskit().draw("mpl", fold=40, filename=f"drawings/qlsa-gates-{iteration}.png")

    draw_encoding(enc, level="ladders"
                  ).to_qiskit().draw("mpl", fold=40, filename=f"drawings/enc-ladders-{iteration}.png")

    plt.close("all")

def phys_est_qlsa(alg, enc, prob):
    """Print physical resource estimates for a superconducting-device preset."""
    phys_est = PhysicalResourceEstimator(preset("superconducting_1e-3")
                                         ).estimate(alg, enc, prob.b, amplitude_encoding)
    print(phys_est.summary())
    print(f"Physical qubits: {phys_est.physical_qubits}")
    print(f"Runtime per attempt (seconds): {phys_est.runtime_s}")
    print(f"Expected runtime (seconds): {phys_est.expected_runtime_s}")

def ratio(var_vec, delta_var_vec):
    """Ratio test; return the largest step up to one that keeps decreasing variables nonnegative."""
    decreasing = delta_var_vec < 0
    if not np.any(decreasing):
        return 1.0
    return min(1.0, np.min(-var_vec[decreasing] / delta_var_vec[decreasing]))