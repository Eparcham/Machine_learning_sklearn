import numpy as np
import matplotlib.pyplot as plt

# 1. Define PDF
def f(x):
    return -0.75 * x * (x - 2)  # only valid for 0 <= x <= 2

# 2. Define domain
x_vals = np.linspace(0, 2, 1000)
pdf_vals = f(x_vals)

# 3. Approximate CDF using trapezoidal rule (numerical integration)
cdf_vals = np.cumsum(pdf_vals) * (x_vals[1] - x_vals[0])  # ∆x = step size
cdf_vals /= cdf_vals[-1]  # Normalize CDF to make sure last value is 1

# 4. Plot PDF and CDF
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, pdf_vals, label='PDF f(x)', color='blue')
plt.title("PDF: f(x)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vals, cdf_vals, label='CDF F(x)', color='green')
plt.title("CDF: F(x)")
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 5. Function to approximate CDF at specific x
def approx_cdf(x_query):
    idx = np.searchsorted(x_vals, x_query)
    if idx >= len(cdf_vals):
        return 1.0
    return cdf_vals[idx]

# 6. Calculate specific probabilities
P1 = approx_cdf(1.5)              # P(X <= 1.5)
P2 = approx_cdf(1.0) - approx_cdf(0.5)  # P(0.5 <= X < 1)
P3 = 1.0 - approx_cdf(1.0)        # P(X >= 1)

print(f"P(X <= 1.5) ≈ {P1:.4f}")
print(f"P(0.5 <= X < 1) ≈ {P2:.4f}")
print(f"P(X >= 1) ≈ {P3:.4f}")
