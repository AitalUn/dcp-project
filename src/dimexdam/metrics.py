"""Простые дескрипторы для сравнения с ground truth: доля фазы и двухточечная функция S2(r)."""
import numpy as np


def phase_fraction(vol: np.ndarray, phase: int = 0) -> float:
    return float((vol == phase).mean())


def s2(vol: np.ndarray, axis: int, phase: int = 0, rmax: int = 32) -> np.ndarray:
    """S2(r) = P(обе точки на расстоянии r вдоль оси axis лежат в фазе phase).
    Считается через FFT, граница периодическая."""
    ind = (vol == phase).astype(np.float32)
    f = np.fft.rfft(ind, axis=axis)
    ac = np.fft.irfft(f * np.conj(f), n=ind.shape[axis], axis=axis) / ind.shape[axis]
    return ac.mean(axis=tuple(a for a in range(3) if a != axis))[:rmax]


def random_subvolumes(vol: np.ndarray, n: int, size: int, seed=0) -> list[np.ndarray]:
    """Случайные кубы size³ из ground-truth керна: эталон для сравнения со сгенерированными."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i, j, k = (rng.integers(0, s - size + 1) for s in vol.shape)
        out.append(vol[i:i + size, j:j + size, k:k + size])
    return out
