import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian(mu, Sigma, title):
    # شبکه دوبعدی
    x, y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    Z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            Z[i, j] = multivariate_gaussian_pdf([x[i, j], y[i, j]], mu, Sigma)
    plt.contourf(x, y, Z, levels=30, cmap='viridis')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar(label='Density')

def multivariate_gaussian_pdf(x, mu, Sigma):
    """
    محاسبه pdf نرمال چندمتغیره در نقطه x
    x: بردار (n,)
    mu: میانگین (n,)
    Sigma: ماتریس کوواریانس (n, n)
    خروجی: مقدار چگالی در نقطه x
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    d = x.shape[-1]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    norm_const = 1.0 / (np.power(2 * np.pi, d/2) * np.sqrt(Sigma_det))
    x_mu = x - mu
    result = np.exp(-0.5 * np.dot(x_mu.T, np.dot(Sigma_inv, x_mu)))
    return norm_const * result


plt.figure(figsize=(18, 5))

# حالت دایره‌ای (ایزوتروپیک)
mu = [0, 0]
Sigma_circ = [[1, 0], [0, 1]]
plt.subplot(1, 3, 1)
plot_gaussian(mu, Sigma_circ, "Circular (Isotropic)\nΣ = I")

# حالت ستونی
Sigma_col = [[1, 0], [0, 0.1]]
plt.subplot(1, 3, 2)
plot_gaussian(mu, Sigma_col, "Column (Elongated x1)\nΣ = diag(1, 0.1)")

# حالت قطری (غیرایزوتروپیک)
Sigma_diag = [[2, 0], [0, 1]]
plt.subplot(1, 3, 3)
plot_gaussian(mu, Sigma_diag, "Diagonal (Anisotropic)\nΣ = diag(2, 1)")

plt.tight_layout()
plt.show()