from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

import wandb
from evaluator.base_evaluator import BaseEvaluator


class CausalExactMatchEvaluator(BaseEvaluator):
    """Evaluator for causal predictions using exact match on formatted causal change descriptions."""

    def __init__(
        self, config: Dict[str, Any], model: Any, tokenizer: Any, test_dataloader: Any, fs_dataloader: Optional[Any] = None, **kwargs: Any
    ) -> None:
        super().__init__(config, model, tokenizer, test_dataloader, fs_dataloader)
        self.main_metric = "exact_match"
        self.greater_is_better = True
        self.generate_limit = config.get(
            "generate_limit", 50
        )  # Allow longer generations for causal descriptions

    def log_metrics(self, prefix: str = "", step: Optional[int] = None) -> None:
        # Overall metrics
        log_metrics = {
            prefix
            + "exact_match": (
                self.metrics["exact_match"] / self.metrics["total"]
                if self.metrics["total"] > 0
                else 0
            ),
            prefix
            + "exact_match_se": self._calculate_standard_error(
                self.metrics["exact_match"], self.metrics["total"]
            ),
            prefix
            + "semantic_match": (
                self.metrics["semantic_match"] / self.metrics["total"]
                if self.metrics["total"] > 0
                else 0
            ),
            prefix
            + "semantic_match_se": self._calculate_standard_error(
                self.metrics["semantic_match"], self.metrics["total"]
            ),
            prefix
            + "valid": (
                self.metrics["valid"] / self.metrics["total"]
                if self.metrics["total"] > 0
                else 0
            ),
            prefix
            + "valid_se": self._calculate_standard_error(
                self.metrics["valid"], self.metrics["total"]
            ),
            prefix + "is_changed_matrix": self.metrics["is_changed_matrix"],
            prefix + "is_changed_precision": self.metrics["is_changed_precision"],
            prefix + "is_changed_recall": self.metrics["is_changed_recall"],
            prefix + "is_changed_f1": self.metrics["is_changed_f1"],
            prefix
            + "is_changed_f1_se": self._calculate_f1_standard_error(
                self.metrics["is_changed_matrix"][0][0],  # tp
                self.metrics["is_changed_matrix"][1][0],  # fp
                self.metrics["is_changed_matrix"][0][1],  # fn
            ),
            prefix + "is_unchanged_precision": self.metrics["is_unchanged_precision"],
            prefix + "is_unchanged_recall": self.metrics["is_unchanged_recall"],
            prefix + "is_unchanged_f1": self.metrics["is_unchanged_f1"],
            prefix
            + "is_unchanged_f1_se": self._calculate_f1_standard_error(
                self.metrics["is_changed_matrix"][1][1],  # tp for unchanged
                self.metrics["is_changed_matrix"][0][1],  # fp for unchanged
                self.metrics["is_changed_matrix"][1][0],  # fn for unchanged
            ),
        }

        # Stratified metrics for changed interventions
        if self.metrics["changed"]["total"] > 0:
            log_metrics.update(
                {
                    prefix
                    + "changed_exact_match": (
                        self.metrics["changed"]["exact_match"]
                        / self.metrics["changed"]["total"]
                    ),
                    prefix
                    + "changed_exact_match_se": self._calculate_standard_error(
                        self.metrics["changed"]["exact_match"],
                        self.metrics["changed"]["total"],
                    ),
                    prefix
                    + "changed_semantic_match": (
                        self.metrics["changed"]["semantic_match"]
                        / self.metrics["changed"]["total"]
                    ),
                    prefix
                    + "changed_semantic_match_se": self._calculate_standard_error(
                        self.metrics["changed"]["semantic_match"],
                        self.metrics["changed"]["total"],
                    ),
                    prefix
                    + "changed_valid": (
                        self.metrics["changed"]["valid"]
                        / self.metrics["changed"]["total"]
                    ),
                    prefix
                    + "changed_valid_se": self._calculate_standard_error(
                        self.metrics["changed"]["valid"],
                        self.metrics["changed"]["total"],
                    ),
                }
            )

        # Stratified metrics for unchanged interventions
        if self.metrics["unchanged"]["total"] > 0:
            log_metrics.update(
                {
                    prefix
                    + "unchanged_exact_match": (
                        self.metrics["unchanged"]["exact_match"]
                        / self.metrics["unchanged"]["total"]
                    ),
                    prefix
                    + "unchanged_exact_match_se": self._calculate_standard_error(
                        self.metrics["unchanged"]["exact_match"],
                        self.metrics["unchanged"]["total"],
                    ),
                    prefix
                    + "unchanged_semantic_match": (
                        self.metrics["unchanged"]["semantic_match"]
                        / self.metrics["unchanged"]["total"]
                    ),
                    prefix
                    + "unchanged_semantic_match_se": self._calculate_standard_error(
                        self.metrics["unchanged"]["semantic_match"],
                        self.metrics["unchanged"]["total"],
                    ),
                    prefix
                    + "unchanged_valid": (
                        self.metrics["unchanged"]["valid"]
                        / self.metrics["unchanged"]["total"]
                    ),
                    prefix
                    + "unchanged_valid_se": self._calculate_standard_error(
                        self.metrics["unchanged"]["valid"],
                        self.metrics["unchanged"]["total"],
                    ),
                }
            )

        wandb.log(log_metrics, step=step)

    def print_metrics(self, prefix: str = "") -> None:
        # Overall metrics
        exact_match = (
            self.metrics["exact_match"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )
        exact_match_se = self._calculate_standard_error(
            self.metrics["exact_match"], self.metrics["total"]
        )
        semantic_match = (
            self.metrics["semantic_match"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )
        semantic_match_se = self._calculate_standard_error(
            self.metrics["semantic_match"], self.metrics["total"]
        )
        valid = (
            self.metrics["valid"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )
        valid_se = self._calculate_standard_error(
            self.metrics["valid"], self.metrics["total"]
        )

        # Calculate F1 standard errors
        is_changed_f1_se = self._calculate_f1_standard_error(
            self.metrics["is_changed_matrix"][0][0],  # tp
            self.metrics["is_changed_matrix"][1][0],  # fp
            self.metrics["is_changed_matrix"][0][1],  # fn
        )
        is_unchanged_f1_se = self._calculate_f1_standard_error(
            self.metrics["is_changed_matrix"][1][1],  # tp for unchanged
            self.metrics["is_changed_matrix"][0][1],  # fp for unchanged
            self.metrics["is_changed_matrix"][1][0],  # fn for unchanged
        )

        print(
            f"{prefix}Overall - Exact Match: {exact_match:.4f} (±{exact_match_se:.4f}), Semantic Match: {semantic_match:.4f} (±{semantic_match_se:.4f}), Valid: {valid:.4f} (±{valid_se:.4f}), Is Changed Matrix: {self.metrics['is_changed_matrix']}, Is Changed Precision: {self.metrics['is_changed_precision']}, Is Changed Recall: {self.metrics['is_changed_recall']}, Is Changed F1: {self.metrics['is_changed_f1']:.4f} (±{is_changed_f1_se:.4f}), Is Unchanged Precision: {self.metrics['is_unchanged_precision']}, Is Unchanged Recall: {self.metrics['is_unchanged_recall']}, Is Unchanged F1: {self.metrics['is_unchanged_f1']:.4f} (±{is_unchanged_f1_se:.4f})"
        )

        # Stratified metrics for changed interventions
        if self.metrics["changed"]["total"] > 0:
            changed_exact = (
                self.metrics["changed"]["exact_match"]
                / self.metrics["changed"]["total"]
            )
            changed_exact_se = self._calculate_standard_error(
                self.metrics["changed"]["exact_match"], self.metrics["changed"]["total"]
            )
            changed_semantic = (
                self.metrics["changed"]["semantic_match"]
                / self.metrics["changed"]["total"]
            )
            changed_semantic_se = self._calculate_standard_error(
                self.metrics["changed"]["semantic_match"],
                self.metrics["changed"]["total"],
            )
            changed_valid = (
                self.metrics["changed"]["valid"] / self.metrics["changed"]["total"]
            )
            changed_valid_se = self._calculate_standard_error(
                self.metrics["changed"]["valid"], self.metrics["changed"]["total"]
            )
            print(
                f"{prefix}Changed ({self.metrics['changed']['total']}) - Exact Match: {changed_exact:.4f} (±{changed_exact_se:.4f}), Semantic Match: {changed_semantic:.4f} (±{changed_semantic_se:.4f}), Valid: {changed_valid:.4f} (±{changed_valid_se:.4f})"
            )

        # Stratified metrics for unchanged interventions
        if self.metrics["unchanged"]["total"] > 0:
            unchanged_exact = (
                self.metrics["unchanged"]["exact_match"]
                / self.metrics["unchanged"]["total"]
            )
            unchanged_exact_se = self._calculate_standard_error(
                self.metrics["unchanged"]["exact_match"],
                self.metrics["unchanged"]["total"],
            )
            unchanged_semantic = (
                self.metrics["unchanged"]["semantic_match"]
                / self.metrics["unchanged"]["total"]
            )
            unchanged_semantic_se = self._calculate_standard_error(
                self.metrics["unchanged"]["semantic_match"],
                self.metrics["unchanged"]["total"],
            )
            unchanged_valid = (
                self.metrics["unchanged"]["valid"] / self.metrics["unchanged"]["total"]
            )
            unchanged_valid_se = self._calculate_standard_error(
                self.metrics["unchanged"]["valid"], self.metrics["unchanged"]["total"]
            )
            print(
                f"{prefix}Unchanged ({self.metrics['unchanged']['total']}) - Exact Match: {unchanged_exact:.4f} (±{unchanged_exact_se:.4f}), Semantic Match: {unchanged_semantic:.4f} (±{unchanged_semantic_se:.4f}), Valid: {unchanged_valid:.4f} (±{unchanged_valid_se:.4f})"
            )

    def _calculate_standard_error(self, successes: int, total: int) -> float:
        """Calculate standard error for a binomial proportion."""
        if total == 0:
            return 0.0
        p = successes / total
        return math.sqrt(p * (1 - p) / total)

    def _calculate_f1_standard_error(self, tp: int, fp: int, fn: int) -> float:
        """Calculate standard error for F1 score using delta method approximation.

        Args:
            tp: True positives
            fp: False positives
            fn: False negatives

        Returns:
            Standard error of F1 score
        """
        if tp == 0:
            return 0.0

        # Total sample size
        n = tp + fp + fn
        if n == 0:
            return 0.0

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision == 0 or recall == 0:
            return 0.0

        # Delta method approximation for F1 standard error
        # Treating tp, fp, fn as multinomial with probabilities p_tp, p_fp, p_fn
        p_tp = tp / n
        p_fp = fp / n
        p_fn = fn / n

        # Partial derivatives of F1 with respect to tp, fp, fn
        denom = (tp + fp) * (tp + fn) * (precision + recall)
        if denom == 0:
            return 0.0

        df_dtp = 4 * fp * fn / (denom**2) * (tp + fp + tp + fn)
        df_dfp = -4 * tp * fn / (denom**2) * (tp + fn)
        df_dfn = -4 * tp * fp / (denom**2) * (tp + fp)

        # Variance using delta method with multinomial covariances
        var_f1 = (
            df_dtp**2 * p_tp * (1 - p_tp)
            + df_dfp**2 * p_fp * (1 - p_fp)
            + df_dfn**2 * p_fn * (1 - p_fn)
            - 2 * df_dtp * df_dfp * p_tp * p_fp
            - 2 * df_dtp * df_dfn * p_tp * p_fn
            - 2 * df_dfp * df_dfn * p_fp * p_fn
        ) / n

        return math.sqrt(max(0, var_f1))

    def pbar_update(self, pbar: Any) -> None:
        exact_match = (
            self.metrics["exact_match"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )
        semantic_match = (
            self.metrics["semantic_match"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )
        valid = (
            self.metrics["valid"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0
        )

        # Add counts for changed/unchanged
        changed_count = self.metrics["changed"]["total"]
        unchanged_count = self.metrics["unchanged"]["total"]

        pbar.set_description(
            f"EM: {exact_match:.3f}, SM: {semantic_match:.3f}, Valid: {valid:.3f} | C: {changed_count}, U: {unchanged_count}"
        )

    def reset_metrics(self) -> None:
        self.metrics = {
            "exact_match": 0,
            "semantic_match": 0,
            # "from_match": 0,
            # "to_match": 0,
            "total": 0,
            "valid": 0,
            "is_changed_matrix": [[0, 0], [0, 0]],
            "is_changed_precision": 0,
            "is_changed_recall": 0,
            "is_changed_f1": 0,
            "is_unchanged_precision": 0,
            "is_unchanged_recall": 0,
            "is_unchanged_f1": 0,
            # Stratified by is_different
            "changed": {
                "exact_match": 0,
                "semantic_match": 0,
                "total": 0,
                "valid": 0,
            },
            "unchanged": {
                "exact_match": 0,
                "semantic_match": 0,
                "total": 0,
                "valid": 0,
            },
        }

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and punctuation."""
        # Remove trailing eos
        text = text.rstrip(self.tokenizer.eos_token)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())
        # Remove trailing punctuation
        text = text.rstrip(".")
        return text.lower()

    def _extract_change_content(
        self, text: str, task: str, original_response: Optional[str] = None
    ) -> Optional[Tuple[str, bool]]:
        """Extract the 'from X to Y' content from causal change descriptions.
        content: str, is_changed: bool
        """
        unchanged_pattern = r".*remain\s+unchanged\s+from\s+<<<([^>]*(?:>[^>]*)*?)>>>"
        match = re.search(unchanged_pattern, text.lower())
        if match:
            return match.group(1).strip(), False

        changed_pattern = r".*change\s+to\s+<<<([^>]*(?:>[^>]*)*?)>>>"
        match = re.search(changed_pattern, text.lower())
        if match:
            return match.group(1).strip(), True
        return None

    def _update_metrics(
        self,
        exact_match: bool,
        semantic_match: bool,
        is_changed_match: bool,
        is_valid: bool,
        pred_is_changed: Optional[bool],
        exp_is_changed: Optional[bool],
        strat_key: Optional[str],
    ) -> None:
        """Update metrics based on evaluation results."""
        if is_valid:
            self.metrics["valid"] += 1
            if strat_key:
                self.metrics[strat_key]["valid"] += 1

            if exact_match:
                self.metrics["exact_match"] += 1
                if strat_key:
                    self.metrics[strat_key]["exact_match"] += 1

            if semantic_match:
                self.metrics["semantic_match"] += 1
                if strat_key:
                    self.metrics[strat_key]["semantic_match"] += 1

            # columns (index 1): pred_changed, pred_unchanged
            # rows (index 0): is_changed, is_unchanged
            self.metrics["is_changed_matrix"][0][0] += int(
                pred_is_changed and exp_is_changed
            )
            self.metrics["is_changed_matrix"][1][0] += int(
                pred_is_changed and not exp_is_changed
            )
            self.metrics["is_changed_matrix"][0][1] += int(
                not pred_is_changed and exp_is_changed
            )
            self.metrics["is_changed_matrix"][1][1] += int(
                not pred_is_changed and not exp_is_changed
            )

            # Calculate precision, recall, and F1 for "is_changed" prediction
            if (
                self.metrics["is_changed_matrix"][0][0]
                + self.metrics["is_changed_matrix"][1][0]
                > 0
            ):
                self.metrics["is_changed_precision"] = self.metrics[
                    "is_changed_matrix"
                ][0][0] / (
                    self.metrics["is_changed_matrix"][0][0]
                    + self.metrics["is_changed_matrix"][1][0]
                )
            else:
                self.metrics["is_changed_precision"] = 0

            if (
                self.metrics["is_changed_matrix"][0][0]
                + self.metrics["is_changed_matrix"][0][1]
                > 0
            ):
                self.metrics["is_changed_recall"] = self.metrics["is_changed_matrix"][
                    0
                ][0] / (
                    self.metrics["is_changed_matrix"][0][0]
                    + self.metrics["is_changed_matrix"][0][1]
                )
            else:
                self.metrics["is_changed_recall"] = 0

            if (
                self.metrics["is_changed_precision"] + self.metrics["is_changed_recall"]
                > 0
            ):
                self.metrics["is_changed_f1"] = (
                    2
                    * self.metrics["is_changed_precision"]
                    * self.metrics["is_changed_recall"]
                    / (
                        self.metrics["is_changed_precision"]
                        + self.metrics["is_changed_recall"]
                    )
                )
            else:
                self.metrics["is_changed_f1"] = 0

            # Calculate precision, recall, and F1 for "is_unchanged" prediction
            if (
                self.metrics["is_changed_matrix"][1][1]
                + self.metrics["is_changed_matrix"][0][1]
                > 0
            ):
                self.metrics["is_unchanged_precision"] = self.metrics[
                    "is_changed_matrix"
                ][1][1] / (
                    self.metrics["is_changed_matrix"][1][1]
                    + self.metrics["is_changed_matrix"][0][1]
                )
            else:
                self.metrics["is_unchanged_precision"] = 0

            if (
                self.metrics["is_changed_matrix"][1][1]
                + self.metrics["is_changed_matrix"][1][0]
                > 0
            ):
                self.metrics["is_unchanged_recall"] = self.metrics["is_changed_matrix"][
                    1
                ][1] / (
                    self.metrics["is_changed_matrix"][1][1]
                    + self.metrics["is_changed_matrix"][1][0]
                )
            else:
                self.metrics["is_unchanged_recall"] = 0

            if (
                self.metrics["is_unchanged_precision"]
                + self.metrics["is_unchanged_recall"]
                > 0
            ):
                self.metrics["is_unchanged_f1"] = (
                    2
                    * self.metrics["is_unchanged_precision"]
                    * self.metrics["is_unchanged_recall"]
                    / (
                        self.metrics["is_unchanged_precision"]
                        + self.metrics["is_unchanged_recall"]
                    )
                )
            else:
                self.metrics["is_unchanged_f1"] = 0

    def evaluate_item(self, predicted_label: str, expected_label: str, full_item: Dict[str, Any]) -> Dict[str, Any]:
        # Get the is_different flag from the item's extra_args
        is_different = full_item.get("extra_args", {}).get("is_different", None)

        # Update overall totals
        self.metrics["total"] += 1

        # Update stratified totals
        if is_different is not None:
            if is_different:
                self.metrics["changed"]["total"] += 1
                strat_key = "changed"
            else:
                self.metrics["unchanged"]["total"] += 1
                strat_key = "unchanged"
        else:
            strat_key = None

        is_valid = False

        # Exact match check
        if full_item["question_type"] in [
            "orig_match",
            "change_match",
        ]:
            import re

            def first_token(text):
                # Split on any sequence of whitespace or punctuation, take the first non-empty result
                tokens = [
                    tok
                    for tok in re.split(
                        r"[\s\.\,\!\?\-\;\:\(\)\[\]\{\}\"\'\/\\\>]+", text.strip()
                    )
                    if tok
                ]
                return tokens[0] if tokens else ""

            predicted_label = first_token(predicted_label)
            expected_label = first_token(expected_label)
            exact_match = predicted_label == expected_label
            is_valid = False
            is_changed_match = full_item["extra_args"].get("is_different", None)
            pred_is_changed = None
            exp_is_changed = None
            semantic_match = exact_match
        else:
            predicted_label = predicted_label.strip().lower().strip(".")
            expected_label = expected_label.strip().lower().strip(".")
            #  Check if change content matches
            pred_change = self._extract_change_content(
                predicted_label,
                full_item["task"],
                full_item["extra_args"].get("original_response", None),
            )
            exp_change = self._extract_change_content(
                expected_label,
                full_item["task"],
                full_item["extra_args"].get("original_response", None),
            )
            if pred_change and exp_change:
                is_valid = True
                pred_content, pred_is_changed = pred_change
                exp_content, exp_is_changed = exp_change

                is_changed_match = pred_is_changed == exp_is_changed
                semantic_match = pred_content == exp_content
                exact_match = (
                    predicted_label == expected_label
                    and pred_is_changed == exp_is_changed
                )
            else:
                exact_match = predicted_label == expected_label
                is_valid = False
                semantic_match = exact_match
                is_changed_match = False
                pred_is_changed = None
                exp_is_changed = None

        # Update all metrics
        self._update_metrics(
            exact_match,
            semantic_match,
            is_changed_match,
            is_valid,
            # is_different,
            pred_is_changed,
            exp_is_changed,
            strat_key,
        )

        # Print results
        diff_status = (
            f" ({'Changed' if is_different else 'Unchanged'})"
            if is_different is not None
            else ""
        )
        print("=" * 60)
        prompt_without_answer = full_item["prompt_without_answer"]
        print(f"Input: {prompt_without_answer}")
        print(
            f"Layer: {full_item['extra_args'].get('layer', 'N/A')}, Feature: {full_item['extra_args'].get('feature_idx', 'N/A')}"
        )
        if (
            "description" in full_item["extra_args"]
            and len(full_item["extra_args"]["description"]) > 0
        ):
            print(f"Feature: {full_item['extra_args']['description']}")
        print(f"\033[94mPredicted: {predicted_label}\033[0m")
        print(f"\033[92mExpected:  {expected_label}\033[0m")
        print(f"\033[93mExact Match: {exact_match}\033[0m")
        print(f"\033[93mSemantic Match: {semantic_match}\033[0m")
        print(f"\033[93mValid: {is_valid}\033[0m")
        print(f"\033[93mIs Changed Matrix: {self.metrics['is_changed_matrix']}\033[0m")
        print(
            f"\033[93mIs Changed Precision: {self.metrics['is_changed_precision']}\033[0m"
        )
        print(f"\033[93mIs Changed Recall: {self.metrics['is_changed_recall']}\033[0m")
        print(f"\033[93mIs Changed F1: {self.metrics['is_changed_f1']}\033[0m")
        print(
            f"\033[93mIs Unchanged Precision: {self.metrics['is_unchanged_precision']}\033[0m"
        )
        print(
            f"\033[93mIs Unchanged Recall: {self.metrics['is_unchanged_recall']}\033[0m"
        )
        print(f"\033[93mIs Unchanged F1: {self.metrics['is_unchanged_f1']}\033[0m")
        print(f"\033[95m{full_item['question_type']}{diff_status}\033[0m")
        print("=" * 60)
        return {
            "exact_match": exact_match,
            "semantic_match": exact_match or semantic_match,
            "valid": is_valid,
            "is_changed_matrix": self.metrics["is_changed_matrix"],
            "is_changed_precision": self.metrics["is_changed_precision"],
            "is_changed_recall": self.metrics["is_changed_recall"],
            "is_changed_f1": self.metrics["is_changed_f1"],
            "is_unchanged_precision": self.metrics["is_unchanged_precision"],
            "is_unchanged_recall": self.metrics["is_unchanged_recall"],
            "is_unchanged_f1": self.metrics["is_unchanged_f1"],
        }

    def finalize_metrics(self) -> None:
        # Metrics are already accumulated, no normalization needed
        pass
