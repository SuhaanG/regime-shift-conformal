"""
Split conformal prediction for lead-time estimation.

Standard split conformal prediction: use a held-out calibration set
(distinct from training) to compute nonconformity scores (absolute
residuals between predicted and true lead time), take the empirical
(1-alpha) quantile of those scores, and construct prediction intervals
of the form [prediction - q, prediction + q] that are guaranteed to
achieve (1-alpha) coverage on exchangeable data.

IMPORTANT CAVEAT, consistent with the recurring-event framing developed
in this paper's related work: standard conformal prediction assumes
exchangeability between calibration and test data. Financial time
series exhibit non-stationarity (the very phenomenon this paper is
about), so the coverage guarantee here is technically valid under the
assumption that calibration and test windows are drawn from a
sufficiently similar distribution -- this assumption and its limits
should be discussed explicitly in the paper's methodology and
limitations sections, not silently assumed.

Only windows with a genuine observed future shift (non-placeholder lead
time) are used for calibration and evaluation, consistent with how the
regressor was trained.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from calibration.lead_time_model import LeadTimeRegressor
from calibration.train_lead_time_model import build_pooled_splits, ALL_CSVS, PLACEHOLDER_LEAD_TIME, HORIZON


def get_predictions_and_targets(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for windows, _, lead_times in dataloader:
            windows = windows.to(device)
            preds = model(windows)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lead_times.numpy())

    return np.array(all_preds), np.array(all_targets)


def compute_nonconformity_scores(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Absolute residual nonconformity score, the standard choice for
    split conformal regression."""
    return np.abs(preds - targets)


def compute_conformal_quantile(nonconformity_scores: np.ndarray, alpha: float = 0.1) -> float:
    """
    Computes the (1-alpha) empirical quantile of nonconformity scores,
    using the standard finite-sample correction (ceil((n+1)(1-alpha))/n)
    that gives the exact coverage guarantee for split conformal
    prediction, rather than a naive quantile.
    """
    n = len(nonconformity_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)  # clip in case of small n
    return np.quantile(nonconformity_scores, q_level)


def evaluate_coverage(preds: np.ndarray, targets: np.ndarray, q: float) -> dict:
    """
    Checks empirical coverage of the constructed intervals
    [pred - q, pred + q] against the nominal target, on a separate
    test set. This is the actual validation that the calibration
    procedure works as claimed.
    """
    lower = preds - q
    upper = preds + q
    covered = (targets >= lower) & (targets <= upper)
    empirical_coverage = np.mean(covered)
    avg_interval_width = 2 * q  # constant width for standard split conformal

    return {
        "empirical_coverage": empirical_coverage,
        "interval_width": avg_interval_width,
        "n_test": len(targets),
    }


def run_conformal_calibration(alpha: float = 0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target coverage: {1 - alpha:.0%} (alpha={alpha})")

    # Rebuild the same pooled splits used in training, so calibration
    # uses the validation set and final evaluation uses the untouched
    # test set -- train/cal/test are kept fully separate.
    train_ds, cal_ds, test_ds = build_pooled_splits(ALL_CSVS)

    n_features = train_ds[0][0].shape[1]
    model = LeadTimeRegressor(input_channels=n_features).to(device)
    model.load_state_dict(torch.load("lead_time_model_checkpoint.pt", map_location=device))

    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- Calibration step ---
    cal_preds, cal_targets = get_predictions_and_targets(model, cal_loader, device)
    valid_cal_mask = cal_targets < PLACEHOLDER_LEAD_TIME
    cal_preds_valid = cal_preds[valid_cal_mask]
    cal_targets_valid = cal_targets[valid_cal_mask]
    print(f"Calibration set: {len(cal_targets_valid)} valid windows (out of {len(cal_targets)} total)")

    nonconformity_scores = compute_nonconformity_scores(cal_preds_valid, cal_targets_valid)
    q = compute_conformal_quantile(nonconformity_scores, alpha=alpha)
    print(f"Conformal quantile (q): {q:.3f} days")

    # --- Test-set evaluation (the actual validation of the guarantee) ---
    test_preds, test_targets = get_predictions_and_targets(model, test_loader, device)
    valid_test_mask = test_targets < PLACEHOLDER_LEAD_TIME
    test_preds_valid = test_preds[valid_test_mask]
    test_targets_valid = test_targets[valid_test_mask]
    print(f"Test set: {len(test_targets_valid)} valid windows (out of {len(test_targets)} total)")

    coverage_results = evaluate_coverage(test_preds_valid, test_targets_valid, q)

    print(f"\n--- Conformal Calibration Results ---")
    print(f"Nominal target coverage: {1 - alpha:.1%}")
    print(f"Empirical test-set coverage: {coverage_results['empirical_coverage']:.1%}")
    print(f"Interval width: ±{q:.2f} days (total width {coverage_results['interval_width']:.2f} days)")
    print(f"Test windows evaluated: {coverage_results['n_test']}")

    if abs(coverage_results["empirical_coverage"] - (1 - alpha)) > 0.05:
        print(
            "\nWARNING: empirical coverage deviates from nominal target by more than 5 "
            "percentage points. This may indicate a distribution shift between "
            "calibration and test sets (e.g., non-stationarity), violating the "
            "exchangeability assumption. Discuss this explicitly in the paper."
        )

    return q, coverage_results


if __name__ == "__main__":
    run_conformal_calibration(alpha=0.1)