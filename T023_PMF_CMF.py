import numpy as np
import matplotlib.pyplot as plt

# PMF data
x_vals = np.array([0, 1, 2, 3, 4])
pmf_vals = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

# Confirm total probability
print(f"Total probability = {pmf_vals.sum():.4f}")

# Cumulative Mass Function (CMF)
cmf_vals = np.cumsum(pmf_vals)

# Plotting
plt.figure(figsize=(12, 5))

# PMF
plt.subplot(1, 2, 1)
plt.stem(x_vals, pmf_vals, basefmt=" ")  # ✅ no 'use_line_collection'
plt.title("PMF: P(X = x)")
plt.xlabel("x")
plt.ylabel("Probability")
plt.grid(True)

# CMF
plt.subplot(1, 2, 2)
plt.step(x_vals, cmf_vals, where='post', color='green')
plt.title("CMF (CDF): P(X ≤ x)")
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.grid(True)

plt.tight_layout()
plt.show()

# Table output
print("\nValues:")
for x, p, c in zip(x_vals, pmf_vals, cmf_vals):
    print(f"P(X = {x}) = {p:.2f}   |   P(X ≤ {x}) = {c:.2f}")
