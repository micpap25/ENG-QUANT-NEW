# IPM Utility Functions
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qrisp.operators import QubitOperator


def draw_lcu_block_encoding_schematic(
    operator: QubitOperator, alpha: float, system_qubits: int, filename: str
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    terms = list(operator.hermitize().to_pauli().terms_dict.items())
    index_qubits = (len(terms) - 1).bit_length()
    wire_count = index_qubits + system_qubits
    wire_y = [wire_count - 1 - i for i in range(wire_count)]

    branch_lines = []
    for i, (term, coefficient) in enumerate(terms):
        # Qrisp's LCU decomposition uses the real coefficients of the hermitized operator.
        coefficient = float(np.real(coefficient))
        signed_term = f"{'-' if coefficient < 0 else ''}{term}"
        branch_lines.append(f"|{i}>: {signed_term}  (weight {abs(coefficient):.3g})")

    fig_height = max(4.0, 0.42 * wire_count + 1.8, 0.3 * len(branch_lines) + 2.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_axis_off()

    for i, y in enumerate(wire_y):
        register = "index" if i < index_qubits else "system"
        register_index = i if i < index_qubits else i - index_qubits
        ax.text(0.65, y, f"{register}[{register_index}]", ha="right", va="center")
        ax.plot([0.8, 7.9], [y, y], color="black", linewidth=0.9)

    def add_box(
        x: float, width: float, rows: list[int], label: str, dashed: bool = False
    ) -> None:
        ys = [wire_y[row] for row in rows]
        low = min(ys) - 0.28
        height = max(0.56, max(ys) - min(ys) + 0.56)
        box = FancyBboxPatch(
            (x, low),
            width,
            height,
            boxstyle="round,pad=0.03",
            facecolor="white",
            edgecolor="black",
            linestyle="--" if dashed else "-",
            linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, low + height / 2, label, ha="center", va="center", fontsize=9)

    all_rows = list(range(wire_count))
    if index_qubits:
        index_rows = list(range(index_qubits))
        add_box(1.05, 1.25, index_rows, "PREP^dagger")
        add_box(4.95, 1.15, index_rows, "PREP")
        add_box(6.45, 1.3, index_rows, "postselect\nindex = 0", dashed=True)
    add_box(2.75, 1.75, all_rows, f"SELECT\n{len(terms)} Pauli terms")

    title_y = wire_count + 0.55
    ax.text(0.8, title_y, "Block encoding from array: LCU structure", fontsize=12, weight="bold")
    ax.text(8.25, title_y, "SELECT branches", fontsize=10, weight="bold", va="center")
    for i, line in enumerate(branch_lines):
        ax.text(8.25, title_y - 0.45 - 0.32 * i, line, fontsize=8, va="top")

    ax.text(
        0.8,
        -0.65,
        f"Left-to-right: PREP^dagger -> SELECT -> PREP;  PREP|0> = sum_j sqrt(w_j/alpha)|j>, w_j=|c_j|\n"
        f"<0|U_A|0> = (A + A^dagger)/(2 alpha),  alpha = {alpha:.4g}",
        fontsize=8,
        va="top",
    )
    ax.set_xlim(0, 12.5)
    ax.set_ylim(min(-1.5, title_y - 0.55 - 0.32 * len(branch_lines)), title_y + 0.9)
    fig.tight_layout()
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


# Ratio Test
def ratio(var_vec, delta_var_vec):
    decreasing = delta_var_vec < 0
    if not np.any(decreasing):
        return 1.0
    return min(1.0, np.min(-var_vec[decreasing] / delta_var_vec[decreasing]))