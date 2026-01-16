"""Functions for stuff"""
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
    Bin(acc=0.00834046, conf=0.00610245, size=85472),
    Bin(acc=0.15806323, conf=0.14380600, size=2377),
    Bin(acc=0.26013772, conf=0.24469279, size=1291),
    Bin(acc=0.38037486, conf=0.34832118, size=932),
    Bin(acc=0.42204301, conf=0.44782602, size=759),
    Bin(acc=0.56355932, conf=0.54866985, size=680),
    Bin(acc=0.59011628, conf=0.65035141, size=676),
    Bin(acc=0.69358974, conf=0.75087687, size=749),
    Bin(acc=0.79605263, conf=0.85360305, size=1098),
    Bin(acc=0.95987707, conf=0.98035512, size=5894),
]


def abs_conf_diff(bucket: Bin) -> float:
    """Calculate the absolute difference between accuracy and confidence for a bin.

    Args:
        bucket: A calibration bin containing accuracy and confidence values.

    Returns:
        The absolute difference |accuracy - confidence|, indicating calibration error.
    """
    return math.fabs(bucket.acc - bucket.conf)


def bin_weight(buckets: list[Bin]) -> Callable[[Bin], float]:
    """Create a weighting function based on bin size relative to total samples.

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

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The maximum |accuracy - confidence| across all bins.

    Raises:
        ValueError: If the input bucket list is empty.
    """
    return max(map(abs_conf_diff, buckets))


def calculate_ece(buckets: list[Bin]) -> float:
    """Calculate the Expected Calibration Error (ECE).

    ECE is a weighted average of the absolute differences between accuracy and confidence
    across all bins, where weights are based on the number of samples in each bin.

    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The ECE value representing overall calibration error.
    """
    total_size = sum(bucket.size for bucket in buckets)
    weighted_errors = [
        (bucket.size / total_size) * abs_conf_diff(bucket) for bucket in buckets
    ]
    return sum(weighted_errors)


def sq_conf_diff(bucket: Bin) -> float:
    """Calculate the squared difference between accuracy and confidence for a bin.
    
    This is the core mathematical component for RMSCE.
    """
    return (bucket.acc - bucket.conf) ** 2


def calculate_rmsce(buckets: list[Bin]) -> float:
    """Calculate the Root Mean Square Calibration Error (RMSCE).

    RMSCE is the square root of the weighted average of squared differences 
    between accuracy and confidence. It penalizes larger deviations more 
    heavily than ECE.
    The formula for RMSCE is:
    
    RMSCE = sqrt( Σ ( |Bin| / N ) * (acc(Bi) - conf(Bin))² )

    Where:
        - |Bi| is the number of samples in bin i.
        - N is the total number of samples across all bins.
        - acc(Bi) is the actual accuracy of bin i.
        - conf(Bi) is the average predicted confidence of bin i
    Args:
        buckets: A list of calibration bins to evaluate.

    Returns:
        The RMSCE value representing weighted quadratic calibration error.
    
    Raises:
        ZeroDivisionError: If the input bucket list is empty.
    """
    weight_func = bin_weight(buckets)
    def weighted_sq_error(b):
        return weight_func(b) * (abs_conf_diff(b) ** 2)
    return math.sqrt(sum(map(weighted_sq_error, buckets)))


def calculate_nll(buckets: list[Bin]) -> float:
    """Calculate the Negative Log-Likelihood (NLL) for calibration bins.

    NLL measures how well the predicted probabilities align with actual outcomes,
    penalizing overconfident incorrect predictions.

    NLL = -1/N * Σ [y * log(p) + (1-y) * log(1-p)]

    Args:
        buckets: A list of calibration bins to evaluate.
    
    Returns:
        The NLL value representing the calibration quality.
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

# --- Testing with your BINS data ---
rmsce_error = calculate_rmsce(BINS)
ece_error = calculate_ece(BINS)
nll_error = calculate_nll(BINS)
mce_error = calculate_mce(BINS)
print(f"RMSCE: {rmsce_error:.8f}\n"
      f"ECE  : {ece_error:.8f}\n"
      f"NLL  : {nll_error:.8f}\n"
      f"MCE  : {mce_error:.8f}\n")
