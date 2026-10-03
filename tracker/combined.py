"""Mode 2: combine the fundamental view (what to buy) with the technical view (when to buy)."""
from __future__ import annotations

from dataclasses import dataclass

from .fundamentals import FundamentalReport
from .strategy import StrategyParams

FUND_TIERS = ("Strong", "Fair", "Weak")
TECH_TIERS = ("Bearish", "Neutral", "Bullish")

# (fundamental tier, technical tier) -> (verdict, tone, explanation)
MATRIX: dict[tuple[str, str], tuple[str, str, str]] = {
    ("Strong", "Bullish"): ("STRONG BUY", "good",
        "A quality business and the chart confirms buyers are in control. This is the textbook setup: "
        "good company, good timing."),
    ("Strong", "Neutral"): ("ACCUMULATE", "good",
        "Solid fundamentals, but the chart is undecided. Long-term investors might buy in small pieces "
        "or wait for a technical trigger."),
    ("Strong", "Bearish"): ("WAIT", "warn",
        "Good company, bad timing. Price is in a downtrend - fundamentals rarely stop a falling price "
        "in the short run. Wait for the trend to turn."),
    ("Fair", "Bullish"): ("BUY (MOMENTUM)", "ok",
        "Average fundamentals with a strong chart. Tradeable on momentum, but there is less of a "
        "quality cushion if the trend breaks - keep stops tight."),
    ("Fair", "Neutral"): ("HOLD", "neutral",
        "Nothing stands out on either side. No edge - holders can hold, new money can look elsewhere."),
    ("Fair", "Bearish"): ("AVOID FOR NOW", "warn",
        "Average business in a downtrend. Little reason to step in until the chart improves."),
    ("Weak", "Bullish"): ("SPECULATIVE", "warn",
        "Price is rising but the business is weak. Momentum can run a long way, but this is a trade, "
        "not an investment - high risk if the story cracks."),
    ("Weak", "Neutral"): ("AVOID", "bad",
        "Weak fundamentals and no technical tailwind."),
    ("Weak", "Bearish"): ("STRONG AVOID", "bad",
        "Weak business and a falling price - both views agree to stay away."),
}


def tech_tier(score: float, p: StrategyParams) -> str:
    if score >= p.buy_threshold:
        return "Bullish"
    if score <= p.sell_threshold:
        return "Bearish"
    return "Neutral"


@dataclass
class CombinedVerdict:
    label: str
    tone: str                    # good / ok / neutral / warn / bad
    explanation: str
    combined_score: float        # 0..100
    fund_tier: str | None
    tech_tier: str
    fund_score: float | None     # 0..100
    tech_score: float            # -100..100


def combine(report: FundamentalReport | None, tech_score: float, p: StrategyParams,
            fund_weight: float = 0.6) -> CombinedVerdict:
    t_tier = tech_tier(tech_score, p)
    tech_0_100 = (tech_score + 100) / 2
    if report is None or not report.available:
        label = {"Bullish": "BUY", "Neutral": "HOLD", "Bearish": "SELL"}[t_tier]
        tone = {"Bullish": "ok", "Neutral": "neutral", "Bearish": "warn"}[t_tier]
        why = (report.note if report is not None else "Fundamentals are not loaded.") + \
            " The verdict falls back to the technical signal alone."
        return CombinedVerdict(f"{label} (technical only)", tone, why, tech_0_100, None, t_tier, None, tech_score)

    label, tone, why = MATRIX[(report.tier, t_tier)]
    combined = fund_weight * report.overall + (1 - fund_weight) * tech_0_100
    return CombinedVerdict(label, tone, why, combined, report.tier, t_tier, report.overall, tech_score)
