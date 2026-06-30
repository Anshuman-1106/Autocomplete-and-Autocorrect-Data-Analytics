"""
User Experience module (Step 6: Assess impact through feedback and surveys).

Since real user feedback isn't available in this environment, this module
simulates a lightweight UX survey: it scores each engine on responsiveness
(from measured latency) and usefulness (from measured accuracy), then
combines them into a simple satisfaction score out of 5 -- standing in for
what a real survey/feedback pipeline would collect from live users.

In a production project, replace `simulate_survey_score` with actual
collected feedback (e.g. thumbs up/down on suggestions, post-task
satisfaction surveys, or A/B test conversion/acceptance rates).
"""

from typing import Dict


def simulate_survey_score(accuracy: float, latency_ms: float) -> float:
    """
    Combine accuracy and responsiveness into an approximate 1-5
    satisfaction score. Users tend to tolerate slower responses if
    accuracy is high, and accuracy matters more than raw speed --
    so accuracy is weighted higher.
    """
    accuracy_component = accuracy * 4.0          # up to 4 points
    # Latency penalty: scores drop as latency grows past 5ms,
    # capped so it never goes negative.
    latency_component = max(0.0, 1.0 - min(latency_ms, 5.0) / 5.0)  # up to 1 point
    return round(accuracy_component + latency_component, 2)


def build_ux_report(autocomplete_results: Dict[str, Dict],
                     autocorrect_results: Dict[str, Dict]) -> Dict[str, float]:
    """Produce a simulated 'user satisfaction' score (out of 5) per engine."""
    report = {}
    for name, res in autocomplete_results.items():
        report[name] = simulate_survey_score(res["top_k_accuracy"], res["avg_latency_ms"])
    for name, res in autocorrect_results.items():
        report[name] = simulate_survey_score(res["accuracy"], res["avg_latency_ms"])
    return report
