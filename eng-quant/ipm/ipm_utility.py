# IPM Utility Functions
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from qlsas import ResourceEstimator, amplitude_encoding, draw, draw_encoding
from qlsas.ft import PhysicalResourceEstimator, preset

def print_qlsa_stats(alg, enc, prob):
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
    phys_est = PhysicalResourceEstimator(preset("superconducting_1e-3")
                                         ).estimate(alg, enc, prob.b, amplitude_encoding)
    print(phys_est.summary())
    print(f"Physical qubits: {phys_est.physical_qubits}")
    print(f"Runtime per attempt (seconds): {phys_est.runtime_s}")
    print(f"Expected runtime (seconds): {phys_est.expected_runtime_s}")

# Ratio Test
def ratio(var_vec, delta_var_vec):
    decreasing = delta_var_vec < 0
    if not np.any(decreasing):
        return 1.0
    return min(1.0, np.min(-var_vec[decreasing] / delta_var_vec[decreasing]))