"""
Reports conformal calibration results across multiple confidence levels
(alpha values), as is standard practice in conformal prediction papers.
Reuses the already-trained lead_time_model_checkpoint.pt -- no
retraining needed, since only the calibration quantile changes with
alpha, not the underlying model.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from calibration.lead_time_model import LeadTimeRegressor
from calibration.train_lead_time_model import build_pooled_splits, ALL_CSVS, PLACEHOLDER_LEAD_TIME
from calibration.conformal_calibration import (
    get_predictions_and_targets,
    compute_nonconformity_scores,
    compute_conformal_quantile,
    evaluate_coverage,
)

ALPHAS = [0.20, 0.10, 0.05]  # corresponding to 80%, 90%, 95% target coverage


def run_multi_alpha_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, cal_ds, test_ds = build_pooled_splits(ALL_CSVS)
    n_features = train_ds[0][0].shape[1]
    model = LeadTimeRegressor(input_channels=n_features).to(device)
    model.load_state_dict(torch.load("lead_time_model_checkpoint.pt", map_location=device))

    cal_loader = DataLoader(cal_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    cal_preds, cal_targets = get_predictions_and_targets(model, cal_loader, device)
    valid_cal_mask = cal_targets < PLACEHOLDER_LEAD_TIME
    cal_preds_valid, cal_targets_valid = cal_preds[valid_cal_mask], cal_targets[valid_cal_mask]

    test_preds, test_targets = get_predictions_and_targets(model, test_loader, device)
    valid_test_mask = test_targets < PLACEHOLDER_LEAD_TIME
    test_preds_valid, test_targets_valid = test_preds[valid_test_mask], test_targets[valid_test_mask]

    nonconformity_scores = compute_nonconformity_scores(cal_preds_valid, cal_targets_valid)

    print(f"{'Nominal Coverage':>18} {'Empirical Coverage':>20} {'Interval Width (±days)':>24}")
    results = []
    for alpha in ALPHAS:
        q = compute_conformal_quantile(nonconformity_scores, alpha=alpha)
        coverage_results = evaluate_coverage(test_preds_valid, test_targets_valid, q)
        results.append({"alpha": alpha, "q": q, **coverage_results})
        print(
            f"{1 - alpha:>17.0%} {coverage_results['empirical_coverage']:>19.1%} {q:>23.2f}"
        )

    return results


if __name__ == "__main__":
    run_multi_alpha_sweep()