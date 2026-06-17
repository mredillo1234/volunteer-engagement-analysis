#!/usr/bin/env python3
"""
volunteer_engagement_analysis.py
================================

A self-contained data pipeline that CLEANS messy volunteer-program data and
ANALYZES it for engagement and retention insights.

Why this matters
----------------
Nonprofits run on volunteer time, but their data is usually a mess: typos,
duplicate sign-ups, dates entered five different ways, blank fields. Before
anyone can answer "are we keeping our volunteers?" the data has to be cleaned.
This script does both halves of that job and then narrates what it found in
plain language -- no spreadsheet-staring required.

The analysis is framed with industrial-organizational (I/O) psychology ideas:
  - retention / churn  (are people still active?)
  - tenure cohorts     (do newer vs. longer-term volunteers behave differently?)
  - engagement         (how much are people actually contributing?)

How to run
----------
    python3 volunteer_engagement_analysis.py
        -> generates a realistic, intentionally-messy sample dataset,
           cleans it, analyzes it, and prints an insights report.

    python3 volunteer_engagement_analysis.py your_data.csv
        -> runs the same pipeline on your own CSV (same column names).

Outputs
-------
    cleaned_volunteers.csv   the tidy, analysis-ready dataset
    engagement_chart.png     a simple visual summary (if matplotlib is present)

Author: built as a portfolio artifact. Heavily commented on purpose so the
logic is easy to walk through in an interview.
"""

import sys
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# A fixed "today" so the analysis is reproducible no matter when it runs.
TODAY = pd.Timestamp("2026-06-17")

# A volunteer counts as "churned" if they haven't been active in this many days.
CHURN_THRESHOLD_DAYS = 90


# ---------------------------------------------------------------------------
# 1. SAMPLE DATA
#    Build a dataset that looks like a real nonprofit export -- on purpose
#    full of the problems you'd actually have to fix.
# ---------------------------------------------------------------------------
def make_messy_sample(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Generate a realistic, deliberately-dirty volunteer dataset."""
    rng = np.random.default_rng(seed)

    first_names = ["maria", "James", "  Aisha", "Wei ", "Sofia", "noah",
                   "Liam", "EMMA", "Diego", "Yuki", "Omar", "grace"]
    last_names = ["Garcia", "smith", "Khan ", " Chen", "Lopez", "OKAFOR",
                  "Patel", "Nguyen", "Rossi", "Adams", "kim", "Silva"]

    # Region typed inconsistently the way humans actually type it.
    region_variants = ["North", "north", "N.", "NORTH ", "South", "south",
                       "S.", "East", "east", "West", " west", "Central", "central"]

    role_variants = ["Tutor", "tutor", "Mentor", "mentor ", "Driver",
                     "Event Staff", "event staff", "Fundraiser", "fundraiser"]

    status_variants = ["Active", "active", "ACTIVE", "Inactive", "inactive",
                       "on hold", "On Hold", ""]

    rows = []
    for i in range(n):
        # Sign-up date somewhere in the last ~3 years.
        signup = TODAY - timedelta(days=int(rng.integers(20, 1100)))

        # Last active date sits somewhere between signup and today.
        # Skew it toward "recent" so the dataset looks like a real program
        # (most people still around, a meaningful minority lapsed).
        window = max((TODAY - signup).days, 1)
        frac = rng.beta(5, 2)  # beta(5,2) clusters near 1.0 -> recent activity
        last_active = signup + timedelta(days=int(window * frac))
        if last_active > TODAY:
            last_active = TODAY

        # Dates stored in several different string formats (the classic mess).
        date_fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%m/%d/%y"])
        signup_str = signup.strftime(date_fmt)
        last_str = last_active.strftime(rng.choice(["%Y-%m-%d", "%m/%d/%Y"]))

        # Hours: mostly fine, but some blanks, some negatives, some as text.
        hours = float(rng.gamma(2.0, 12.0))
        hours_cell = round(hours, 1)
        roll = rng.random()
        if roll < 0.06:
            hours_cell = ""                       # missing
        elif roll < 0.09:
            hours_cell = -abs(hours_cell)          # data-entry error
        elif roll < 0.12:
            hours_cell = f"{hours_cell} hrs"       # text contamination

        fn = rng.choice(first_names)
        ln = rng.choice(last_names)
        name = f"{fn} {ln}"

        # Email: some valid, some missing, some malformed.
        eroll = rng.random()
        if eroll < 0.08:
            email = ""
        elif eroll < 0.13:
            email = f"{fn.strip().lower()}.at.email.com"   # missing @
        else:
            email = f"{fn.strip().lower()}.{ln.strip().lower()}@example.org"

        rows.append({
            "volunteer_id": 1000 + i,
            "name": name,
            "email": email,
            "region": rng.choice(region_variants),
            "role": rng.choice(role_variants),
            "signup_date": signup_str,
            "last_active_date": last_str,
            "hours_logged": hours_cell,
            "status": rng.choice(status_variants),
        })

    df = pd.DataFrame(rows)

    # Inject ~20 duplicate rows (same volunteer entered twice).
    dupes = df.sample(20, random_state=seed).copy()
    df = pd.concat([df, dupes], ignore_index=True)

    # Shuffle so duplicates aren't sitting next to each other.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. CLEANING HELPERS
#    Small, single-purpose functions. Easy to test, easy to explain.
# ---------------------------------------------------------------------------
def clean_text(value):
    """Trim whitespace and collapse internal double-spaces; title-case names."""
    if pd.isna(value):
        return np.nan
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned else np.nan


def standardize_region(value):
    """Map every messy spelling of a region onto one canonical label."""
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower().rstrip(".")
    mapping = {
        "north": "North", "n": "North",
        "south": "South", "s": "South",
        "east": "East", "e": "East",
        "west": "West", "w": "West",
        "central": "Central", "c": "Central",
    }
    return mapping.get(v, "Unknown")


def standardize_status(value):
    """Collapse status variants into Active / Inactive / On Hold / Unknown."""
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower()
    if v == "active":
        return "Active"
    if v == "inactive":
        return "Inactive"
    if v in ("on hold", "onhold", "hold"):
        return "On Hold"
    return "Unknown"


def parse_hours(value):
    """Pull a clean, non-negative number of hours out of a messy cell."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    # Grab the first number that appears (handles "12.5 hrs" etc.).
    match = re.search(r"-?\d+(\.\d+)?", str(value))
    if not match:
        return np.nan
    num = float(match.group())
    # A negative hours value is a data-entry error -> treat as missing.
    return abs(num) if num < 0 else num


def parse_date(value):
    """Parse a date string regardless of which common format it's in."""
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce", dayfirst=False)


def is_valid_email(value):
    """Loose check: does this look like a real email address?"""
    if pd.isna(value):
        return False
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(value)))


# ---------------------------------------------------------------------------
# 3. THE CLEANING PIPELINE
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step and return the tidy frame plus a change log."""
    log = {}
    start_rows = len(df)

    # --- de-duplicate on volunteer_id (keep the first sighting) ---
    before = len(df)
    df = df.drop_duplicates(subset="volunteer_id", keep="first").copy()
    log["duplicate_rows_removed"] = before - len(df)

    # --- normalize text fields ---
    df["name"] = df["name"].apply(clean_text).str.title()
    df["email"] = df["email"].apply(clean_text).str.lower()

    # --- standardize categories ---
    df["region"] = df["region"].apply(standardize_region)
    df["role"] = df["role"].apply(clean_text).str.title()
    df["status"] = df["status"].apply(standardize_status)

    # --- parse numbers and dates ---
    df["hours_logged"] = df["hours_logged"].apply(parse_hours)
    df["signup_date"] = df["signup_date"].apply(parse_date)
    df["last_active_date"] = df["last_active_date"].apply(parse_date)

    # --- flag email quality (don't drop, just mark) ---
    df["email_valid"] = df["email"].apply(is_valid_email)
    log["invalid_or_missing_emails"] = int((~df["email_valid"]).sum())

    # --- fill missing hours with 0 (no logged hours == zero contribution) ---
    log["missing_hours_filled"] = int(df["hours_logged"].isna().sum())
    df["hours_logged"] = df["hours_logged"].fillna(0.0)

    # --- derived fields used by the analysis ---
    df["tenure_days"] = (TODAY - df["signup_date"]).dt.days
    df["days_since_active"] = (TODAY - df["last_active_date"]).dt.days
    df["churned"] = df["days_since_active"] > CHURN_THRESHOLD_DAYS

    # Tenure cohort: a plain-language bucket instead of a raw number.
    def cohort(days):
        if pd.isna(days):
            return "Unknown"
        if days < 180:
            return "New (under 6 mo)"
        if days < 365:
            return "Established (6-12 mo)"
        return "Veteran (1 yr+)"

    df["tenure_cohort"] = df["tenure_days"].apply(cohort)

    log["rows_in"] = start_rows
    log["rows_out"] = len(df)
    return df, log


# ---------------------------------------------------------------------------
# 4. ANALYSIS -> PLAIN-LANGUAGE INSIGHTS
#    Every number is wrapped in a sentence that explains what it means.
# ---------------------------------------------------------------------------
def analyze(df: pd.DataFrame, log: dict) -> str:
    """Turn the cleaned data into a readable insights report."""
    lines = []
    add = lines.append

    total = len(df)
    active = int((df["status"] == "Active").sum())
    churned = int(df["churned"].sum())
    retained = total - churned
    retention_rate = retained / total * 100 if total else 0

    add("=" * 64)
    add("VOLUNTEER ENGAGEMENT REPORT")
    add(f"Generated {TODAY.date()}  |  {total} unique volunteers analyzed")
    add("=" * 64)

    # --- data quality recap (shows the cleaning actually did something) ---
    add("\nDATA CLEANING SUMMARY")
    add("-" * 64)
    add(f"Started with {log['rows_in']} rows, ended with {log['rows_out']} "
        f"after removing {log['duplicate_rows_removed']} duplicate sign-ups.")
    add(f"{log['missing_hours_filled']} volunteers had no logged hours "
        f"(treated as zero contribution).")
    add(f"{log['invalid_or_missing_emails']} records had a missing or "
        f"malformed email and were flagged for follow-up.")

    # --- retention headline ---
    add("\nRETENTION (the headline number)")
    add("-" * 64)
    add(f"About {retention_rate:.0f}% of volunteers are still engaged "
        f"(active within the last {CHURN_THRESHOLD_DAYS} days).")
    add(f"That's {retained} retained and {churned} who have gone quiet. "
        f"The quiet group is the one worth a re-engagement campaign.")

    # --- engagement (hours) ---
    hrs = df["hours_logged"]
    add("\nENGAGEMENT (how much people contribute)")
    add("-" * 64)
    add(f"Volunteers have logged {hrs.sum():,.0f} hours in total.")
    add(f"The typical volunteer (the median) has logged about "
        f"{hrs.median():.0f} hours.")
    # The "vital few": what share of hours comes from the top 10%?
    top10_cut = hrs.quantile(0.90)
    top10_share = hrs[hrs >= top10_cut].sum() / hrs.sum() * 100 if hrs.sum() else 0
    add(f"The most active 10% of volunteers account for roughly "
        f"{top10_share:.0f}% of all hours -- a small core carries a lot of "
        f"the load, which is common and a retention risk if they burn out.")

    # --- tenure cohorts: do newer vs older volunteers stay engaged? ---
    add("\nTENURE COHORTS (do newer volunteers behave differently?)")
    add("-" * 64)
    cohort_order = ["New (under 6 mo)", "Established (6-12 mo)",
                    "Veteran (1 yr+)", "Unknown"]
    for c in cohort_order:
        sub = df[df["tenure_cohort"] == c]
        if len(sub) == 0:
            continue
        c_churn = sub["churned"].mean() * 100
        add(f"  {c:<22} {len(sub):>3} people, "
            f"{c_churn:>4.0f}% have gone quiet, "
            f"avg {sub['hours_logged'].mean():>4.0f} hrs each")
    add("Reading this: if a cohort has a high 'gone quiet' rate, that's where "
        "onboarding or check-ins are leaking people.")

    # --- region breakdown ---
    add("\nBY REGION")
    add("-" * 64)
    region_tbl = (df.groupby("region")
                    .agg(volunteers=("volunteer_id", "count"),
                         avg_hours=("hours_logged", "mean"),
                         churn_pct=("churned", "mean"))
                    .sort_values("volunteers", ascending=False))
    for region, r in region_tbl.iterrows():
        add(f"  {region:<10} {int(r['volunteers']):>3} volunteers, "
            f"avg {r['avg_hours']:>4.0f} hrs, "
            f"{r['churn_pct']*100:>4.0f}% gone quiet")

    # --- one clear recommendation ---
    worst_region = region_tbl.sort_values("churn_pct", ascending=False).index[0]
    worst_cohort = (df.groupby("tenure_cohort")["churned"].mean()
                      .drop("Unknown", errors="ignore").idxmax())
    add("\nWHAT TO DO WITH THIS")
    add("-" * 64)
    add(f"1. Re-engage the {churned} quiet volunteers first -- that's the "
        f"fastest win.")
    add(f"2. The '{worst_cohort}' cohort drops off the most; tighten "
        f"check-ins at that stage.")
    add(f"3. The {worst_region} region has the highest quiet rate; it may "
        f"need more local support.")
    add("=" * 64)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. OPTIONAL CHART
# ---------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, path: str = "engagement_chart.png"):
    """Save a simple two-panel visual. Skips quietly if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # no display needed
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: retained vs. churned.
    churned = int(df["churned"].sum())
    retained = len(df) - churned
    ax1.bar(["Still engaged", "Gone quiet"], [retained, churned],
            color=["#4C9F70", "#D1495B"])
    ax1.set_title("Volunteer retention")
    ax1.set_ylabel("Volunteers")

    # Panel 2: average hours by tenure cohort.
    order = ["New (under 6 mo)", "Established (6-12 mo)", "Veteran (1 yr+)"]
    means = [df.loc[df["tenure_cohort"] == c, "hours_logged"].mean()
             for c in order]
    ax2.bar(range(len(order)), means, color="#3D5A80")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(["New", "Established", "Veteran"])
    ax2.set_title("Avg hours by tenure")
    ax2.set_ylabel("Hours")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        source = sys.argv[1]
        print(f"Loading your data from: {source}")
        raw = pd.read_csv(source)
    else:
        print("No file given -> generating a realistic messy sample dataset.\n")
        raw = make_messy_sample()

    cleaned, log = clean(raw)

    # Save the tidy dataset.
    out_csv = "cleaned_volunteers.csv"
    cleaned.to_csv(out_csv, index=False)

    # Print the insights report.
    print(analyze(cleaned, log))

    # Save a chart if we can.
    chart = make_chart(cleaned)

    print(f"\nFiles written:")
    print(f"  - {out_csv}  (cleaned, analysis-ready data)")
    if chart:
        print(f"  - {chart}  (visual summary)")


if __name__ == "__main__":
    main()
