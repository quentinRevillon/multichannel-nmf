"""EM algorithm for multichannel NMF in convolutive mixtures (Ozerov & Fevotte, 2010).

Model, for each frequency bin f and frame t:
    x(f, t) = A(f) s(f, t) + b(f, t),          x: (I,), A(f): (I, J)
    s_j(f, t) = sum_{k in K_j} c_k(f, t),      c_k(f, t) ~ Nc(0, W[f, k] H[k, t])
    b(f, t) ~ Nc(0, diag(sigma_b[f]))
The E-step computes posterior moments of s and c given x; the M-step updates
A, sigma_b, W and H in closed form (Section IV of the paper).

Array shapes: X (F, T, I), W (F, K), H (K, T), A (F, I, J), sigma_b (F, I),
labels (K,) gives the source index of each NMF component.

Usage:
    from mnmf.em import fit
    params, labels, log_liks = fit(X, n_sources=2, n_components=4, n_iter=100)
"""
import numpy as np

EPS = 1e-12


def make_labels(n_sources, n_components):
    """Assign `n_components` consecutive NMF components to each source.

    Returns:
        Int array (n_sources * n_components,).
    """
    return np.repeat(np.arange(n_sources), n_components)


def init_params(X, labels, seed=0):
    """Draw random initial parameters scaled to the mixture power.

    Args:
        X: mixture STFT (F, T, I).
        labels: source index of each component (K,).
        seed: seed of the random generator.

    Returns:
        Dict with keys W (F, K), H (K, T), A (F, I, J), sigma_b (F, I).
    """
    rng = np.random.default_rng(seed)
    F, T, I = X.shape
    K, J = len(labels), labels.max() + 1
    power = np.mean(np.abs(X) ** 2)
    A = rng.standard_normal((F, I, J)) + 1j * rng.standard_normal((F, I, J))
    return {
        "W": rng.random((F, K)) + 0.1,
        "H": (rng.random((K, T)) + 0.1) * power / K,
        "A": A / np.linalg.norm(A, axis=1, keepdims=True),
        "sigma_b": 1e-2 * np.mean(np.abs(X) ** 2, axis=1),
    }


def component_variances(W, H):
    """Return the NMF component variances W[f, k] H[k, t] as (F, T, K)."""
    return np.einsum("fk,kt->ftk", W, H)


def source_variances(Vc, labels):
    """Sum component variances per source: (F, T, K) -> (F, T, J)."""
    onehot = (labels[:, None] == np.arange(labels.max() + 1)).astype(float)
    return Vc @ onehot


def mixture_covariance(A, Vs, sigma_b):
    """Return Sigma_x(f, t) = A(f) diag(Vs(f, t)) A(f)^H + diag(sigma_b(f)), shape (F, T, I, I)."""
    noise = np.einsum("fi,ij->fij", sigma_b, np.eye(sigma_b.shape[1]))
    return np.einsum("fij,ftj,fkj->ftik", A, Vs, A.conj()) + noise[:, None]


def wiener_gain(A, V, Sx_inv):
    """Return the posterior gain diag(V) A^H Sigma_x^{-1}, shape (F, T, J, I).

    Args:
        A: mixing filters (F, I, J), one column per latent signal.
        V: latent variances (F, T, J).
        Sx_inv: inverse mixture covariances (F, T, I, I).
    """
    return V[..., None] * np.einsum("fij,ftik->ftjk", A.conj(), Sx_inv)


def log_likelihood(X, Sigma_x, Sx_inv):
    """Return sum over (f, t) of log Nc(x(f, t); 0, Sigma_x(f, t))."""
    quad = np.einsum("fti,ftik,ftk->ft", X.conj(), Sx_inv, X).real
    _, logdet = np.linalg.slogdet(Sigma_x)
    return float(-np.sum(quad + logdet + X.shape[-1] * np.log(np.pi)))


def source_statistics(X, A, Vs, Sx_inv):
    """Compute the frequency-wise sufficient statistics of the sources.

    Returns:
        R_xx (F, I, I), R_xs (F, I, J), R_ss (F, J, J), averaged over frames.
    """
    G = wiener_gain(A, Vs, Sx_inv)
    s = np.einsum("ftji,fti->ftj", G, X)
    post_cov = np.einsum("ftj,jl->ftjl", Vs, np.eye(Vs.shape[-1])) - np.einsum("ftji,fil,ftl->ftjl", G, A, Vs)
    R_xx = np.einsum("fti,ftk->fik", X, X.conj()) / X.shape[1]
    R_xs = np.einsum("fti,ftj->fij", X, s.conj()) / X.shape[1]
    R_ss = (np.einsum("ftj,ftl->fjl", s, s.conj()) + post_cov.sum(axis=1)) / X.shape[1]
    return R_xx, R_xs, R_ss


def component_powers(X, A, Vc, labels, Sx_inv):
    """Return the posterior power E[|c_k(f, t)|^2 | x] of each component, shape (F, T, K)."""
    A_comp = A[:, :, labels]
    G = wiener_gain(A_comp, Vc, Sx_inv)
    c = np.einsum("ftki,fti->ftk", G, X)
    shrink = np.einsum("ftki,fik->ftk", G, A_comp).real
    return np.abs(c) ** 2 + Vc * (1 - shrink)


def e_step(X, params, labels):
    """Run the E-step.

    Returns:
        (stats, log_lik) where stats holds R_xx, R_xs, R_ss and U (F, T, K),
        and log_lik is the log-likelihood of the current parameters.
    """
    Vc = component_variances(params["W"], params["H"])
    Vs = source_variances(Vc, labels)
    Sigma_x = mixture_covariance(params["A"], Vs, params["sigma_b"])
    Sx_inv = np.linalg.inv(Sigma_x)
    R_xx, R_xs, R_ss = source_statistics(X, params["A"], Vs, Sx_inv)
    U = component_powers(X, params["A"], Vc, labels, Sx_inv)
    stats = {"R_xx": R_xx, "R_xs": R_xs, "R_ss": R_ss, "U": U}
    return stats, log_likelihood(X, Sigma_x, Sx_inv)


def m_step(stats, H):
    """Run the M-step from E-step statistics and the current activations H.

    Returns:
        Dict of updated parameters W, H, A, sigma_b.
    """
    R_xx, R_xs, R_ss, U = stats["R_xx"], stats["R_xs"], stats["R_ss"], stats["U"]
    A = R_xs @ np.linalg.inv(R_ss)
    AR = A @ R_xs.conj().transpose(0, 2, 1)
    noise = R_xx - AR - AR.conj().transpose(0, 2, 1) + A @ R_ss @ A.conj().transpose(0, 2, 1)
    W = np.mean(U / H.T[None], axis=1)
    H = np.mean(U / W[:, None], axis=0).T
    sigma_b = np.maximum(np.diagonal(noise, axis1=1, axis2=2).real, EPS)
    return {"W": W, "H": H, "A": A, "sigma_b": sigma_b}


def normalize(params, labels):
    """Remove scale ambiguities: unit-norm columns of A(f), unit-sum columns of W.

    The likelihood is unchanged: the scales are moved to W, then to H.
    """
    A, W, H = params["A"], params["W"], params["H"]
    norms = np.linalg.norm(A, axis=1)
    W = W * norms[:, labels] ** 2
    scale = W.sum(axis=0)
    return {**params, "A": A / norms[:, None], "W": W / scale, "H": H * scale[:, None]}


def fit(X, n_sources, n_components, n_iter=100, seed=0):
    """Estimate the model parameters with EM.

    Args:
        X: mixture STFT (F, T, I).
        n_sources: number of sources J.
        n_components: number of NMF components per source.
        n_iter: number of EM iterations.
        seed: seed of the random initialisation.

    Returns:
        (params, labels, log_liks) with log_liks the log-likelihood before each iteration.
    """
    labels = make_labels(n_sources, n_components)
    params = init_params(X, labels, seed)
    log_liks = []
    for _ in range(n_iter):
        stats, log_lik = e_step(X, params, labels)
        params = normalize(m_step(stats, params["H"]), labels)
        log_liks.append(log_lik)
    return params, labels, np.array(log_liks)
