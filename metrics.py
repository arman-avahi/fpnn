"""Calibration metrics for evaluating machine learning model confidence.

This module provides functions and utilities for calculating various calibration
metrics that assess how well a model's predicted confidence aligns with its actual
accuracy. Calibration is critical for deploying ML models in high-stakes applications
where uncertainty quantification matters.

The module includes implementations of:
    - ECE (Expected Calibration Error): Weighted average of calibration errors
    - MCE (Maximum Calibration Error): Worst-case calibration error across bins
    - RMSCE (Root Mean Square Calibration Error): Quadratic calibration metric
    - NLL (Negative Log-Likelihood): Probabilistic calibration quality measure

Typical usage example:
    bins = [Bin(acc=0.9, conf=0.85, size=100), ...]
    ece = calculate_ece(bins)
    mce = calculate_mce(bins)
    rmsce = calculate_rmsce(bins)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math


@dataclass(frozen=True)
class Bin:
    """A calibration bin for evaluating ML model confidence calibration.

    Represents a single bin in a calibration histogram, containing the accuracy,
    confidence, and number of samples for predictions falling within a specific
    probability range. Used for computing calibration metrics like ECE and RMSCE.

    Attributes:
        acc: The accuracy of predictions in this bin (fraction of correct predictions).
        conf: The mean predicted confidence/probability for samples in this bin.
        size: The number of samples that fall into this bin.
    """
    acc: float
    conf: float
    size: int

BINS: list[Bin] = [
    Bin(acc=0.00909378, conf=0.0059244, size=85553),
    Bin(acc=0.17035775, conf=0.14248397, size=2348),
    Bin(acc=0.2745098, conf=0.24646743, size=1275),
    Bin(acc=0.34451902, conf=0.34819741, size=894),
    Bin(acc=0.41820768, conf=0.44746627, size=703),
    Bin(acc=0.54585799, conf=0.55071687, size=676),
    Bin(acc=0.5862069, conf=0.65021489, size=696),
    Bin(acc=0.68848168, conf=0.74971831, size=764),
    Bin(acc=0.78737233, conf=0.85490657, size=1077),
    Bin(acc=0.95084153, conf=0.98102668, size=6001),
]

def abs_conf_diff(bucket: Bin) -> float:
    """Calculate the absolute difference between accuracy and confidence for a bin.

    Acceptance criteria:
        1. Determinism - the same bucket yields the same result.
        2. No mutation - the input bin is not modified (frozen dataclass).
        3. Non-negative output - returns |acc - conf| >= 0.

    Args:
        bucket: A calibration bin containing accuracy and confidence values.

    Returns:
        The absolute difference |accuracy - confidence|, indicating calibration error.
    """
    return math.fabs(bucket.acc - bucket.conf)


def bin_weight(buckets: list[Bin]) -> Callable[[Bin], float]:
    """Create a weighting function based on bin size relative to total samples.

    Acceptance criteria:
        1. Determinism - the same buckets yield a function with consistent behavior.
        2. No mutation - the input list is not modified.
        3. Valid weights - returned weights are in [0, 1] and sum to 1 across all bins.
        4. Closure capture - the returned function captures total size from buckets.

    Args:
        buckets: A list of calibration bins used to calculate total sample count.

    Returns:
        A function that takes a Bin and returns its weight (size / total).
    """
    def weight(bucket: Bin) -> float:
        return bucket.size / sum(map(lambda b: b.size, buckets))
    return weight


def calculate_mce(buckets: list[Bin]) -> float:
    """Calculate the Maximum Calibration Error (MCE).

    MCE is the largest absolute difference between accuracy and confidence
    across all bins, representing the worst-case calibration error.

    MCE = max( |acc_i - conf_i|) ) over all bins

    Acceptance criteria:
        1. Determinism - the same buckets yield the same MCE.
        2. No mutation - the input list is not modified.
        3. Maximum selection - returns the largest calibration gap across all bins.
        4. Non-negative output - MCE >= 0.
        5. Validation - raises ValueError if buckets is empty.

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The maximum |accuracy - confidence| across all bins.
    
    Raises:
        ValueError: If the input list is empty
    """
    return max(map(abs_conf_diff, buckets))


def calculate_ece(buckets: list[Bin]) -> float:
    """Calculate the Expected Calibration Error (ECE).

    ECE is a weighted average of the absolute differences between accuracy and confidence
    across all bins, where weights are based on the number of samples in each bin.

    ECE = Σₘ (|Bₘ| / N) · |accₘ − confₘ|
    
    Acceptance criteria:
        1. Determinism - the same buckets yield the same ECE.
        2. No mutation - the input list is not modified.
        3. Proper weighting - each bin's error is weighted by its sample proportion.
        4. Non-negative output - ECE >= 0.
        5. Bounded - ECE <= 1 for valid calibration bins.

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The ECE value representing overall calibration error.
    
    Raises:
        ValueError: If the input list is empty
    """
    if len(buckets) == 0:
        raise ValueError("Empty bucket list")
    total_size = sum(bucket.size for bucket in buckets)
    weighted_errors = [
        (bucket.size / total_size) * abs_conf_diff(bucket) for bucket in buckets
    ]
    return sum(weighted_errors)


def calculate_rmsce(buckets: list[Bin]) -> float:
    """Calculate the Root Mean Square Calibration Error (RMSCE).

    RMSCE is the square root of the weighted average of squared differences
    between accuracy and confidence. It penalizes larger deviations more
    heavily than ECE.
    The formula for RMSCE is:

    RMSCE = √( Σₘ (|Bₘ| / N) · (accₘ − confₘ)² )

    Where:
        - |Bi| is the number of samples in bin i.
        - N is the total number of samples across all bins.
        - acc(Bi) is the actual accuracy of bin i.
        - conf(Bi) is the average predicted confidence of bin i

    Acceptance criteria:
        1. Determinism - the same buckets yield the same RMSCE.
        2. No mutation - the input list is not modified.
        3. Proper weighting - squared errors are weighted by sample proportion.
        4. Non-negative output - RMSCE >= 0.
        5. Quadratic penalty - larger errors contribute disproportionately more.
        6. Validation - raises ZeroDivisionError if buckets is empty.

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The RMSCE value representing weighted quadratic calibration error.

    Raises:
        ZeroDivisionError: If the input bucket list is empty.
    """
    if len(buckets) == 0:
        raise ZeroDivisionError("Cannot calculate RMSCE for empty bucket list")
    weight_func = bin_weight(buckets)
    def weighted_sq_error(bucket):
        return weight_func(bucket) * (abs_conf_diff(bucket) ** 2)
    return math.sqrt(sum(map(weighted_sq_error, buckets)))


def calculate_nll(buckets: list[Bin]) -> float:
    """Calculate the Negative Log-Likelihood (NLL) for calibration bins.

    NLL measures how well the predicted probabilities align with actual outcomes,
    penalizing overconfident incorrect predictions.

    NLL = (1/N) · Σₘ |Bₘ| · [ -accₘ · log(confₘ) - (1 - accₘ) · log(1 - confₘ) ]

    Acceptance criteria:
        1. Determinism - the same buckets yield the same NLL.
        2. No mutation - the input list is not modified.
        3. Proper weighting - each bin's NLL is weighted by sample size.
        4. Numerical stability - clamps probabilities to [1e-15, 1-1e-15] to avoid log(0).
        5. Empty bin handling - filters out bins with size = 0.

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The NLL value representing the calibration quality.

    Raises:
        ZeroDivisionError: if the bins have 0 total samples
    """

    non_empty = [b for b in buckets if b.size > 0]
    total_size = sum(b.size for b in non_empty)
    nll_sum = sum(
        b.size * (
            -b.acc * math.log(max(min(b.conf, 1 - 1e-15), 1e-15)) -
            (1 - b.acc) * math.log(max(min(1 - b.conf, 1 - 1e-15), 1e-15))
        )
        for b in non_empty
    )
    return nll_sum / total_size

if __name__ == "__main__":
    nll_error = calculate_nll(BINS)
    rmsce_error = calculate_rmsce(BINS)
    ece_error = calculate_ece(BINS)
    mce_error = calculate_mce(BINS)
    print(f"RMSCE: {rmsce_error:.8f}\n"
        f"ECE  : {ece_error:.8f}\n"
        f"NLL  : {nll_error:.8f}\n"
        f"MCE  : {mce_error:.8f}\n")
