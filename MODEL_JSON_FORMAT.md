# Model JSON format (library-agnostic)

This repository writes a `model.json` file that fully describes an optimization model in a solver-agnostic way:

- **Variables** (names, types, bounds)
- **Constraints** (each as bounds on an expression)
- **Objectives** (sense + expression)

This document describes the structure and how to **reconstruct the model conceptually** from the JSON without assuming any specific modeling framework.

---

## Top-level structure

`model.json` is a JSON object with:

- **`name`**: string model name
- **`variables`**: object mapping *variable name → variable metadata*
- **`constraints`**: object mapping *constraint name → constraint definition*
- **`objectives`**: object mapping *objective name → objective definition*

---

## Variables

`variables` is a mapping from a **unique variable name** to:

- **`domain`**: string (informational; e.g. `"Reals"`, `"Binary"`, `"Integers"`)
- **`is_binary`**, **`is_integer`**, **`is_continuous`**: booleans (redundant but convenient)
- **`lower`**: number or `null` (missing lower bound = unbounded below)
- **`upper`**: number or `null` (missing upper bound = unbounded above)
- **`fixed`**: boolean (if `true`, treat the variable as fixed to `value`)
- **`value`**: number or `null` (optional; can be absent/uninitialized for unsolved models)

### Conceptual reconstruction

For each variable $x$:

- If `fixed == true` and `value` is a number: impose $x = \text{value}$.
- Otherwise, impose bound constraints:
  - if `lower` is a number: $x \ge \text{lower}$
  - if `upper` is a number: $x \le \text{upper}$
- Use `is_binary` / `is_integer` / `is_continuous` to assign the variable type:
  - `is_binary == true`  → $x \in \{0,1\}$
  - else if `is_integer == true` → $x \in \mathbb{Z}$
  - else → $x \in \mathbb{R}$

---

## Expressions

Both constraints and objectives reference expressions using the same representation.

An expression object has:

- **`constant`**: number
- **`linear`**: list of terms `{ "var": "<var_name>", "coef": <number> }`
- **`quadratic`**: list of terms `{ "var1": "<var_name>", "var2": "<var_name>", "coef": <number> }`
- **`nonlinear`**: `null` or a string (residual expression text if the expression is not purely linear/quadratic)

### Expression semantics

The expression represents a scalar function $f(x)$ of variables:

$$
f(x) = c
 + \sum_{t \in \text{linear}} a_t \, x_{\text{var}(t)}
 + \sum_{q \in \text{quadratic}} b_q \, x_{\text{var1}(q)} \, x_{\text{var2}(q)}
 + \text{(optional nonlinear residual)}
$$

Where:

- $c$ is `constant`
- each linear term adds `coef * variable`
- each quadratic term adds `coef * var1 * var2`

Notes:

- A squared term is represented by `var1 == var2`.
- The quadratic list may contain cross terms (e.g. $x_i x_j$).
- If `nonlinear` is not `null`, the expression contains additional structure not captured by the linear/quadratic lists. In that case, the JSON still preserves the residual as text, but you cannot reliably reconstruct a fully structured expression without interpreting that string.

---

## Constraints

`constraints` is a mapping from a **unique constraint name** to:

- **`lower`**: number or `null`
- **`upper`**: number or `null`
- **`equality`**: boolean
- **`body`**: an expression object (see above)

### Constraint semantics (bounded form)

Every constraint is stored in **bounded form**:

$$
\text{lower} \le \text{body}(x) \le \text{upper}
$$

with the convention:

- if `lower == null`, the lower bound is not present (unbounded below)
- if `upper == null`, the upper bound is not present (unbounded above)

This means constraints are interpreted as:

- **Upper-only** (`lower=null`, `upper=u`): $\text{body}(x) \le u$
- **Lower-only** (`lower=l`, `upper=null`): $\text{body}(x) \ge l$
- **Ranged** (`lower=l`, `upper=u`): $l \le \text{body}(x) \le u$
- **Equality**: typically `equality=true` and `lower` equals `upper`

Importantly:

- The JSON does **not** force constraints into a normalized “$Ax+b \le 0$” form.
- If you need “$\le 0$” form, you can convert:
  - $\text{body}(x) \le u$  ⇔  $\text{body}(x) - u \le 0$
  - $\text{body}(x) \ge l$  ⇔  $l - \text{body}(x) \le 0$
  - $l \le \text{body}(x) \le u$ ⇔ two inequalities:
    - $\text{body}(x) - u \le 0$
    - $l - \text{body}(x) \le 0$

### How to reconstruct constraints programmatically

For each constraint:

1. Build $f(x)$ from `body.constant`, `body.linear`, `body.quadratic`.
2. If `upper` is a number, add inequality $f(x) \le \text{upper}$.
3. If `lower` is a number, add inequality $f(x) \ge \text{lower}$.

If `equality == true` and both bounds exist, you may treat it as:

$$
f(x) = \text{upper} = \text{lower}
$$

---

## Objectives

`objectives` is a mapping from **unique objective name** to:

- **`sense`**: `"minimize"` or `"maximize"`
- **`expr`**: an expression object

### Objective semantics

For each objective:

- Reconstruct $g(x)$ from the expression object.
- Optimize according to `sense`:
  - `"minimize"` → minimize $g(x)$
  - `"maximize"` → maximize $g(x)$

If multiple objectives exist in the JSON, you must decide how to handle them (e.g., pick one, do lexicographic optimization, weighted sum, etc.). The JSON simply records all objectives that were present.

---

## Minimal reconstruction checklist

To rebuild the model conceptually:

1. Create all variables by name with type and bound constraints.
2. For every constraint, interpret it as bounds on `body(x)` and add the corresponding inequality/equality constraints.
3. Pick an objective (or handle multi-objective) and set sense + expression.

This is sufficient to fully recover the mathematical optimization problem represented by the JSON, up to any information contained only in `nonlinear` strings.

