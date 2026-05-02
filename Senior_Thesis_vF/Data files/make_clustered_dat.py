#!/usr/bin/env python3
"""
Generate a single clustered-demand scenario by rewriting only the `param INC := ... ;`
block in an existing AMPL/Pyomo .dat file.

What this script preserves:
- original site assignment for every patient (so site counts stay exactly the same)
- all non-INC content in the .dat file

What this script changes:
- only the arrival day `t` in each nonzero INC record

Example use:
    python make_clustered_dat.py --input Data500_profileA.dat --output Data500_profileA_clustered.dat

Optional custom parameters:
    python make_clustered_dat.py \
        --input Data500_profileA.dat \
        --output Data500_profileA_clustered.dat \
        --seed 42 \
        --time-min 1 \
        --time-max 130 \
        --center1 20 --std1 3.5 --weight1 0.50 \
        --center2 45 --std2 3.5 --weight2 0.30 \
        --center3 70 --std3 3.5 --weight3 0.20
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev


INC_BLOCK_PATTERN = re.compile(
    r"(param\s+INC\s*:=\s*)(.*?)(\s*;\s*)",
    flags=re.DOTALL | re.IGNORECASE
)

INC_LINE_PATTERN = re.compile(
    r"^\s*(p\d+)\s+(c\d+)\s+(\d+)\s+([0-9.]+)\s*$"
)

PATIENT_PATTERN = re.compile(r"p(\d+)$")


def patient_sort_key(pname: str):
    m = PATIENT_PATTERN.match(pname)
    return int(m.group(1)) if m else pname


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def extract_inc_block(dat_text: str):
    m = INC_BLOCK_PATTERN.search(dat_text)
    if not m:
        raise ValueError("Could not find 'param INC := ... ;' block in the .dat file.")

    prefix = m.group(1)
    body = m.group(2)
    suffix = m.group(3)

    records = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        mm = INC_LINE_PATTERN.match(line)
        if mm:
            p, c, t, val = mm.groups()
            val = float(val)
            if abs(val) > 1e-12:
                records.append({
                    "p": p,
                    "c": c,
                    "t": int(t),
                    "val": val,
                })

    if not records:
        raise ValueError("No nonzero INC records found in the .dat file.")

    return prefix, records, suffix


def replace_inc_block(dat_text: str, new_block: str) -> str:
    return INC_BLOCK_PATTERN.sub(new_block, dat_text, count=1)


def weighted_choice(items, weights, rng):
    x = rng.random()
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if x <= cumulative:
            return item
    return items[-1]


def sample_discrete_normal(center, std, lo, hi, rng):
    if std <= 0:
        day = int(round(center))
        return max(lo, min(hi, day))

    while True:
        day = int(round(rng.gauss(center, std)))
        if lo <= day <= hi:
            return day


def assign_clustered_days(records, clusters, time_min, time_max, seed, max_patients_per_day=None):
    """
    Preserve original patient-site assignments exactly.
    Only reassign the arrival day t.
    """
    rng = random.Random(seed)

    # One nonzero INC record per patient in your files
    patient_info = {}
    for r in records:
        patient_info[r["p"]] = {
            "c": r["c"],
            "val": r["val"],
            "original_t": r["t"],
        }

    patients = sorted(patient_info.keys(), key=patient_sort_key)

    weights = [c["weight"] for c in clusters]
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(f"Cluster weights must sum to 1.0, but sum to {weight_sum}.")

    day_counts = Counter()
    new_records = []

    for p in patients:
        chosen_cluster = weighted_choice(clusters, weights, rng)
        day = sample_discrete_normal(
            center=chosen_cluster["center"],
            std=chosen_cluster["std"],
            lo=time_min,
            hi=time_max,
            rng=rng
        )

        if max_patients_per_day is not None and day_counts[day] >= max_patients_per_day:
            found = False
            for radius in range(1, time_max - time_min + 1):
                candidates = []
                if day - radius >= time_min:
                    candidates.append(day - radius)
                if day + radius <= time_max:
                    candidates.append(day + radius)
                rng.shuffle(candidates)

                for cand in candidates:
                    if day_counts[cand] < max_patients_per_day:
                        day = cand
                        found = True
                        break
                if found:
                    break

            if not found:
                raise RuntimeError(
                    f"Could not assign day for patient {p}; day cap too restrictive."
                )

        day_counts[day] += 1
        new_records.append({
            "p": p,
            "c": patient_info[p]["c"],   # preserve original site exactly
            "t": day,
            "val": patient_info[p]["val"]
        })

    return new_records


def format_value(v: float) -> str:
    if abs(v - round(v)) < 1e-12:
        return str(int(round(v)))
    return str(v)


def build_inc_block(prefix: str, records, suffix: str) -> str:
    lines = [prefix]
    for r in records:
        lines.append(f"{r['p']} {r['c']} {r['t']} {format_value(r['val'])}")
    lines.append(suffix.strip())
    return "\n".join(lines) + "\n"


def demand_metrics(records, time_min, time_max):
    arrivals_by_day = Counter(r["t"] for r in records)
    arrivals_by_site = Counter(r["c"] for r in records)

    daily_series = [arrivals_by_day.get(t, 0) for t in range(time_min, time_max + 1)]
    total = sum(daily_series)
    avg_daily = mean(daily_series)
    std_daily = pstdev(daily_series) if len(daily_series) > 1 else 0.0
    cv = std_daily / avg_daily if avg_daily > 0 else 0.0
    peak = max(daily_series) if daily_series else 0
    peak_to_avg = peak / avg_daily if avg_daily > 0 else 0.0

    best_7_sum = 0
    best_7_start = time_min
    for start in range(time_min, time_max - 7 + 2):
        s = sum(arrivals_by_day.get(t, 0) for t in range(start, start + 7))
        if s > best_7_sum:
            best_7_sum = s
            best_7_start = start

    return {
        "total_patients": total,
        "avg_daily": avg_daily,
        "std_daily": std_daily,
        "cv_daily": cv,
        "peak_daily": peak,
        "peak_to_avg": peak_to_avg,
        "arrivals_by_day": arrivals_by_day,
        "arrivals_by_site": arrivals_by_site,
        "best_7_sum": best_7_sum,
        "best_7_start": best_7_start,
    }


def print_summary(title: str, records, time_min, time_max):
    metrics = demand_metrics(records, time_min, time_max)

    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"Total patients:              {metrics['total_patients']}")
    print(f"Average daily arrivals:      {metrics['avg_daily']:.3f}")
    print(f"Std. dev. daily arrivals:    {metrics['std_daily']:.3f}")
    print(f"CV of daily arrivals:        {metrics['cv_daily']:.3f}")
    print(f"Peak daily arrivals:         {metrics['peak_daily']}")
    print(f"Peak-to-average ratio:       {metrics['peak_to_avg']:.3f}")
    print(
        f"Busiest 7-day window:        days {metrics['best_7_start']}"
        f"-{metrics['best_7_start'] + 6} ({metrics['best_7_sum']} arrivals)"
    )

    print("\nSite counts (preserved):")
    for site in sorted(metrics["arrivals_by_site"]):
        print(f"  {site}: {metrics['arrivals_by_site'][site]}")

    print("\nTop 10 busiest days:")
    busiest = sorted(
        metrics["arrivals_by_day"].items(),
        key=lambda kv: (-kv[1], kv[0])
    )[:10]
    for day, count in busiest:
        print(f"  day {day}: {count}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one clustered-demand .dat file by rewriting only the INC block."
    )

    parser.add_argument("--input", required=True, help="Path to the input .dat file.")
    parser.add_argument("--output", required=True, help="Path to the output .dat file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--time-min", type=int, default=1, help="Minimum day index.")
    parser.add_argument("--time-max", type=int, default=130, help="Maximum day index.")
    parser.add_argument("--max-patients-per-day", type=int, default=None,
                        help="Optional cap on arrivals per day.")

    # Three configurable clusters
    parser.add_argument("--center1", type=float, default=20.0)
    parser.add_argument("--std1", type=float, default=3.5)
    parser.add_argument("--weight1", type=float, default=0.50)

    parser.add_argument("--center2", type=float, default=45.0)
    parser.add_argument("--std2", type=float, default=3.5)
    parser.add_argument("--weight2", type=float, default=0.30)

    parser.add_argument("--center3", type=float, default=70.0)
    parser.add_argument("--std3", type=float, default=3.5)
    parser.add_argument("--weight3", type=float, default=0.20)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.time_min > args.time_max:
        raise ValueError("--time-min must be <= --time-max.")

    clusters = [
        {"center": args.center1, "std": args.std1, "weight": args.weight1},
        {"center": args.center2, "std": args.std2, "weight": args.weight2},
        {"center": args.center3, "std": args.std3, "weight": args.weight3},
    ]

    dat_text = read_text(args.input)
    prefix, old_records, suffix = extract_inc_block(dat_text)

    print_summary("ORIGINAL DEMAND SCHEDULE", old_records, args.time_min, args.time_max)

    new_records = assign_clustered_days(
        records=old_records,
        clusters=clusters,
        time_min=args.time_min,
        time_max=args.time_max,
        seed=args.seed,
        max_patients_per_day=args.max_patients_per_day,
    )

    print_summary("NEW CLUSTERED DEMAND SCHEDULE", new_records, args.time_min, args.time_max)

    new_inc_block = build_inc_block(prefix, new_records, suffix)
    new_dat_text = replace_inc_block(dat_text, new_inc_block)

    write_text(args.output, new_dat_text)
    print(f"Wrote clustered .dat file to: {args.output}")


if __name__ == "__main__":
    main()
