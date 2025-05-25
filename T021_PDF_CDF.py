import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the symbolic variable and PDF
x = sp.symbols('x')
f_x = -3/4 * x * (x - 2)  # PDF
f_x = sp.simplify(f_x)

# Step 2: Check normalization
area = sp.integrate(f_x, (x, 0, 2))
print(f"Total area under PDF (should be 1): {area}")

# Step 3: Compute CDF analytically
F_x = sp.integrate(f_x, (x, 0, x))  # Symbolic CDF
F_x = sp.simplify(F_x)
print(f"Symbolic CDF: F(x) = {F_x}")

# Step 4: Plot PDF and CDF
f_lambdified = sp.lambdify(x, f_x, 'numpy')
F_lambdified = sp.lambdify(x, F_x, 'numpy')

x_vals = np.linspace(0, 2, 500)
pdf_vals = f_lambdified(x_vals)
cdf_vals = F_lambdified(x_vals)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, pdf_vals, label="PDF", color='blue')
plt.title("PDF: f(x)")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_vals, cdf_vals, label="CDF", color='green')
plt.title("CDF: F(x)")
plt.xlabel("x")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Step 5: Compute probabilities
# P(X <= 1.5)
P1 = F_x.subs(x, 1.5)

# P(0.5 <= X < 1) = F(1) - F(0.5)
P2 = F_x.subs(x, 1.0) - F_x.subs(x, 0.5)

# P(X >= 1) = 1 - F(1)
P3 = 1 - F_x.subs(x, 1.0)

print(f"P(X <= 1.5) = {P1.evalf():.4f}")
print(f"P(0.5 <= X < 1) = {P2.evalf():.4f}")
print(f"P(X >= 1) = {P3.evalf():.4f}")
