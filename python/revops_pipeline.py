"""
revops_pipeline.py
------------------
Cohere Portfolio — RevOps Analyst Demonstration

Simulates the data work of a Revenue Operations Analyst at a B2B SaaS company
(modelled loosely on an AI company's GTM motion — enterprise deals, expansions,
renewals, churns). Demonstrates:

  1. Synthetic GTM data generation (accounts, opportunities, ARR, product usage)
  2. SQL-style analytics via pandas (mirrors Looker/BigQuery SQL patterns)
  3. Quote-to-cash pipeline metrics
  4. Customer 360 view: CRM + usage + support + marketing attribution
  5. Churn and expansion revenue modelling
  6. Data quality / hygiene checks (a core RevOps responsibility)

SQL queries are written as both pandas operations AND as raw SQL strings
(SQLite-compatible) so the repo demonstrates both competencies.

Usage:
    python revops_pipeline.py                  # generate data + run all analytics
    python revops_pipeline.py --export         # save to CSV
    python revops_pipeline.py --sql            # print raw SQL queries
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# =============================================================================
# Synthetic data generation
# =============================================================================

INDUSTRIES    = ["Finance", "Healthcare", "Telecom", "Retail", "Government", "Legal", "Manufacturing"]
SEGMENTS      = ["SMB", "Mid-Market", "Enterprise", "Strategic"]
STAGES        = ["Prospecting", "Discovery", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
PRODUCTS      = ["Command API", "North Platform", "Embed API", "Rerank API", "Professional Services"]
REGIONS       = ["Toronto", "New York", "San Francisco", "London", "Paris"]
SOURCES       = ["Outbound SDR", "Inbound Demo", "Partner Referral", "Event", "Existing Customer"]
CS_MANAGERS   = ["Angie V.", "Marcus T.", "Priya K.", "Leo C.", "Fatima B."]
CHURN_REASONS = ["Budget cut", "Competitor switch", "Low usage", "M&A", "Product gap"]

rng = random.Random(42)


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def generate_accounts(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        segment   = rng.choice(SEGMENTS)
        arr_base  = {"SMB": 12000, "Mid-Market": 60000, "Enterprise": 250000, "Strategic": 800000}[segment]
        arr       = round(arr_base * rng.uniform(0.6, 1.8), -2)
        created   = random_date(date(2022, 1, 1), date(2024, 6, 1))
        renewal   = created + timedelta(days=365 * rng.randint(1, 3))
        churned   = rng.random() < 0.12
        rows.append({
            "account_id":       f"ACC-{i:04d}",
            "account_name":     f"{rng.choice(['Apex','Nova','Core','Peak','Edge','Vox','Prism'])} {rng.choice(['Labs','Corp','Inc','Group','AI','Tech'])}",
            "industry":         rng.choice(INDUSTRIES),
            "segment":          segment,
            "region":           rng.choice(REGIONS),
            "arr_usd":          arr,
            "contract_start":   created.isoformat(),
            "contract_renewal": renewal.isoformat(),
            "csm":              rng.choice(CS_MANAGERS),
            "primary_product":  rng.choice(PRODUCTS),
            "is_churned":       churned,
            "churn_reason":     rng.choice(CHURN_REASONS) if churned else "",
            "health_score":     rng.randint(20, 40) if churned else rng.randint(55, 99),
        })
    return pd.DataFrame(rows)


def generate_opportunities(accounts: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    rows = []
    account_ids = accounts["account_id"].tolist()
    for i in range(1, n + 1):
        acct_id   = rng.choice(account_ids)
        stage     = rng.choices(STAGES, weights=[10, 15, 20, 15, 30, 10])[0]
        created   = random_date(date(2023, 1, 1), date(2025, 1, 1))
        closed    = created + timedelta(days=rng.randint(14, 180)) if stage.startswith("Closed") else None
        arr       = round(rng.uniform(5000, 500000), -3)
        rows.append({
            "opp_id":        f"OPP-{i:05d}",
            "account_id":    acct_id,
            "opp_name":      f"{rng.choice(['Expansion','New Logo','Renewal','Upsell'])} — {rng.choice(PRODUCTS)}",
            "stage":         stage,
            "arr_usd":       arr,
            "source":        rng.choice(SOURCES),
            "created_date":  created.isoformat(),
            "close_date":    closed.isoformat() if closed else "",
            "days_in_stage": rng.randint(1, 90),
            "is_won":        stage == "Closed Won",
        })
    return pd.DataFrame(rows)


def generate_usage(accounts: pd.DataFrame) -> pd.DataFrame:
    """Monthly API usage per account — last 3 months."""
    rows = []
    months = ["2024-10", "2024-11", "2024-12"]
    for _, acc in accounts.iterrows():
        base_calls = rng.randint(100, 50000)
        for month in months:
            trend = rng.uniform(0.8, 1.3)
            rows.append({
                "account_id":   acc["account_id"],
                "month":        month,
                "api_calls":    int(base_calls * trend),
                "tokens_m":     round(base_calls * trend * rng.uniform(0.001, 0.005), 2),
                "active_users": rng.randint(1, 50),
                "feature_flags": rng.randint(1, 8),
            })
            base_calls = int(base_calls * trend)
    return pd.DataFrame(rows)


def generate_support_tickets(accounts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priorities = ["Low", "Medium", "High", "Critical"]
    for _, acc in accounts.iterrows():
        n_tickets = rng.randint(0, 8)
        for j in range(n_tickets):
            created = random_date(date(2024, 9, 1), date(2024, 12, 31))
            rows.append({
                "ticket_id":    f"TKT-{rng.randint(10000,99999)}",
                "account_id":   acc["account_id"],
                "created_date": created.isoformat(),
                "priority":     rng.choice(priorities),
                "resolved":     rng.random() > 0.1,
                "ttrs_hours":   rng.randint(1, 72),
            })
    return pd.DataFrame(rows)


# =============================================================================
# SQL analytics queries (pandas + raw SQL strings)
# =============================================================================

class RevOpsAnalytics:
    """
    Core RevOps analytics — mirrors what a RevOps Analyst does daily.
    Each method includes both the pandas implementation and
    the equivalent SQL for the docstring.
    """

    def __init__(self, accounts, opportunities, usage, tickets):
        self.accounts      = accounts
        self.opportunities = opportunities
        self.usage         = usage
        self.tickets       = tickets

    # ------------------------------------------------------------------
    # 1. ARR WATERFALL — new, expansion, churn, net
    # ------------------------------------------------------------------
    def arr_waterfall(self) -> pd.DataFrame:
        """
        SQL equivalent:
            SELECT
                segment,
                SUM(CASE WHEN is_churned = 0 THEN arr_usd ELSE 0 END) AS active_arr,
                SUM(CASE WHEN is_churned = 1 THEN arr_usd ELSE 0 END) AS churned_arr,
                COUNT(*) AS account_count,
                AVG(arr_usd) AS avg_arr
            FROM accounts
            GROUP BY segment
            ORDER BY active_arr DESC
        """
        return (
            self.accounts
            .groupby("segment")
            .agg(
                active_arr    = ("arr_usd", lambda x: x[self.accounts.loc[x.index, "is_churned"] == False].sum()),
                churned_arr   = ("arr_usd", lambda x: x[self.accounts.loc[x.index, "is_churned"] == True].sum()),
                account_count = ("account_id", "count"),
                avg_arr       = ("arr_usd", "mean"),
            )
            .round(0)
            .sort_values("active_arr", ascending=False)
            .reset_index()
        )

    # ------------------------------------------------------------------
    # 2. PIPELINE COVERAGE — by stage and source
    # ------------------------------------------------------------------
    def pipeline_coverage(self) -> pd.DataFrame:
        """
        SQL equivalent:
            SELECT
                stage,
                source,
                COUNT(*) AS opp_count,
                SUM(arr_usd) AS total_arr,
                AVG(days_in_stage) AS avg_days_in_stage,
                SUM(CASE WHEN is_won = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM opportunities
            GROUP BY stage, source
            ORDER BY total_arr DESC
        """
        return (
            self.opportunities
            .groupby(["stage", "source"])
            .agg(
                opp_count         = ("opp_id", "count"),
                total_arr         = ("arr_usd", "sum"),
                avg_days_in_stage = ("days_in_stage", "mean"),
                win_rate          = ("is_won", "mean"),
            )
            .round({"total_arr": 0, "avg_days_in_stage": 1, "win_rate": 3})
            .sort_values("total_arr", ascending=False)
            .reset_index()
        )

    # ------------------------------------------------------------------
    # 3. CUSTOMER 360 — CRM + usage + tickets joined
    # ------------------------------------------------------------------
    def customer_360(self) -> pd.DataFrame:
        """
        SQL equivalent:
            SELECT
                a.account_id, a.account_name, a.segment, a.arr_usd,
                a.health_score, a.csm,
                SUM(u.api_calls) AS total_api_calls_3m,
                AVG(u.active_users) AS avg_monthly_users,
                COUNT(t.ticket_id) AS open_tickets,
                AVG(t.ttrs_hours) AS avg_resolution_hours
            FROM accounts a
            LEFT JOIN usage u ON a.account_id = u.account_id
            LEFT JOIN tickets t ON a.account_id = t.account_id AND t.resolved = 0
            GROUP BY a.account_id
            ORDER BY a.arr_usd DESC
        """
        usage_agg = (
            self.usage
            .groupby("account_id")
            .agg(
                total_api_calls_3m = ("api_calls", "sum"),
                avg_monthly_users  = ("active_users", "mean"),
            )
            .round(1)
        )

        ticket_agg = (
            self.tickets[self.tickets["resolved"] == False]
            .groupby("account_id")
            .agg(
                open_tickets          = ("ticket_id", "count"),
                avg_resolution_hours  = ("ttrs_hours", "mean"),
            )
            .round(1)
        )

        c360 = (
            self.accounts
            .merge(usage_agg,  on="account_id", how="left")
            .merge(ticket_agg, on="account_id", how="left")
            .fillna({"open_tickets": 0, "total_api_calls_3m": 0, "avg_monthly_users": 0})
            .sort_values("arr_usd", ascending=False)
        )
        return c360

    # ------------------------------------------------------------------
    # 4. CHURN RISK SCORING — composite risk model
    # ------------------------------------------------------------------
    def churn_risk_model(self) -> pd.DataFrame:
        """
        Risk score = weighted combination of:
            - health_score (inverse)
            - usage trend (calls declining = risk)
            - open critical tickets
            - days to renewal

        SQL equivalent (simplified):
            SELECT account_id, account_name,
                   100 - health_score AS risk_from_health,
                   CASE WHEN usage_trend < 0 THEN 25 ELSE 0 END AS risk_from_usage,
                   (SELECT COUNT(*) FROM tickets WHERE account_id = a.account_id
                    AND priority = 'Critical' AND resolved = 0) * 10 AS risk_from_tickets,
                   DATEDIFF(contract_renewal, CURRENT_DATE) AS days_to_renewal
            FROM accounts a
            ORDER BY total_risk_score DESC
        """
        # Usage trend: compare last month vs first month
        usage_trend = (
            self.usage
            .sort_values("month")
            .groupby("account_id")
            .agg(first_month_calls=("api_calls", "first"),
                 last_month_calls =("api_calls", "last"))
        )
        usage_trend["usage_trend_pct"] = (
            (usage_trend["last_month_calls"] - usage_trend["first_month_calls"])
            / usage_trend["first_month_calls"].replace(0, 1) * 100
        ).round(1)

        # Critical open tickets
        critical_tickets = (
            self.tickets[
                (self.tickets["priority"] == "Critical") &
                (self.tickets["resolved"] == False)
            ]
            .groupby("account_id")
            .size()
            .rename("critical_open_tickets")
        )

        df = (
            self.accounts[~self.accounts["is_churned"]]
            .merge(usage_trend[["usage_trend_pct"]], on="account_id", how="left")
            .merge(critical_tickets, on="account_id", how="left")
            .fillna({"usage_trend_pct": 0, "critical_open_tickets": 0})
        )

        # Composite risk score (0–100)
        df["risk_health"]  = (100 - df["health_score"]) * 0.40
        df["risk_usage"]   = np.where(df["usage_trend_pct"] < -15, 25, 0)
        df["risk_tickets"] = (df["critical_open_tickets"] * 10).clip(0, 25)
        df["risk_score"]   = (df["risk_health"] + df["risk_usage"] + df["risk_tickets"]).round(1)
        df["risk_tier"]    = pd.cut(df["risk_score"], bins=[0, 25, 50, 75, 100],
                                     labels=["Low", "Medium", "High", "Critical"])

        return (
            df[["account_id", "account_name", "segment", "arr_usd",
                "health_score", "usage_trend_pct", "critical_open_tickets",
                "risk_score", "risk_tier", "csm"]]
            .sort_values("risk_score", ascending=False)
        )

    # ------------------------------------------------------------------
    # 5. DATA HYGIENE — quality checks (core RevOps responsibility)
    # ------------------------------------------------------------------
    def data_hygiene_report(self) -> dict:
        """
        Flags data quality issues that would corrupt CRM reporting.
        A clean RevOps system is only as good as its data quality.
        """
        issues = []

        # Opps without close dates in closed stages
        closed_no_date = self.opportunities[
            (self.opportunities["stage"].str.startswith("Closed")) &
            (self.opportunities["close_date"] == "")
        ]
        if len(closed_no_date):
            issues.append({
                "check": "Closed opps missing close_date",
                "count": len(closed_no_date),
                "severity": "High",
                "ids": closed_no_date["opp_id"].head(3).tolist(),
            })

        # Accounts with ARR = 0
        zero_arr = self.accounts[self.accounts["arr_usd"] == 0]
        if len(zero_arr):
            issues.append({"check": "Accounts with $0 ARR", "count": len(zero_arr), "severity": "Medium", "ids": []})

        # Churned accounts without churn reason
        churned_no_reason = self.accounts[
            (self.accounts["is_churned"]) & (self.accounts["churn_reason"] == "")
        ]
        if len(churned_no_reason):
            issues.append({"check": "Churned accounts missing churn_reason", "count": len(churned_no_reason), "severity": "Medium", "ids": []})

        # Duplicate opportunity names per account
        dupes = (
            self.opportunities
            .groupby(["account_id", "opp_name"])
            .size()
            .reset_index(name="count")
        )
        dupe_count = len(dupes[dupes["count"] > 1])
        if dupe_count:
            issues.append({"check": "Duplicate opp names per account", "count": dupe_count, "severity": "Low", "ids": []})

        return {
            "total_issues":   len(issues),
            "high_severity":  sum(1 for i in issues if i["severity"] == "High"),
            "issues":         issues,
            "data_quality_pct": round((1 - len(issues) / 10) * 100, 1),
        }

    def print_report(self):
        """Print a formatted RevOps analytics report."""
        print("\n" + "=" * 70)
        print("  REVOPS ANALYTICS REPORT — Cohere Enterprise GTM")
        print("=" * 70)

        print("\n§1  ARR WATERFALL BY SEGMENT")
        print(self.arr_waterfall().to_string(index=False))

        print("\n§2  TOP PIPELINE BY STAGE × SOURCE (top 10)")
        print(self.pipeline_coverage().head(10).to_string(index=False))

        print("\n§3  CUSTOMER 360 — Top 5 by ARR")
        c360_cols = ["account_name", "segment", "arr_usd", "health_score",
                     "total_api_calls_3m", "open_tickets"]
        print(self.customer_360()[c360_cols].head(5).to_string(index=False))

        print("\n§4  CHURN RISK — Top 10 at-risk accounts")
        risk_cols = ["account_name", "segment", "arr_usd", "risk_score", "risk_tier", "csm"]
        print(self.churn_risk_model()[risk_cols].head(10).to_string(index=False))

        print("\n§5  DATA HYGIENE REPORT")
        hygiene = self.data_hygiene_report()
        print(f"  Data quality score: {hygiene['data_quality_pct']}%")
        print(f"  Total issues: {hygiene['total_issues']}  |  High severity: {hygiene['high_severity']}")
        for issue in hygiene["issues"]:
            print(f"  [{issue['severity']:6}] {issue['check']} — {issue['count']} records")

        print("\n" + "=" * 70)


# =============================================================================
# SQLite demo — shows raw SQL competency
# =============================================================================

RAW_SQL_QUERIES = {
    "net_arr_by_segment": """
        -- Net ARR by segment: active minus churned
        SELECT
            segment,
            SUM(CASE WHEN is_churned = 0 THEN arr_usd ELSE 0 END)  AS active_arr,
            SUM(CASE WHEN is_churned = 1 THEN arr_usd ELSE 0 END)  AS churned_arr,
            SUM(CASE WHEN is_churned = 0 THEN arr_usd ELSE -arr_usd END) AS net_arr,
            COUNT(*) AS accounts
        FROM accounts
        GROUP BY segment
        ORDER BY net_arr DESC
    """,

    "win_rate_by_source": """
        -- Win rate and avg deal size by lead source
        SELECT
            source,
            COUNT(*) AS total_opps,
            SUM(is_won) AS won,
            ROUND(SUM(is_won) * 100.0 / COUNT(*), 1) AS win_rate_pct,
            ROUND(AVG(CASE WHEN is_won = 1 THEN arr_usd END), 0) AS avg_won_arr
        FROM opportunities
        GROUP BY source
        ORDER BY win_rate_pct DESC
    """,

    "churn_by_industry": """
        -- Churn rate and churned ARR by industry
        SELECT
            industry,
            COUNT(*) AS total_accounts,
            SUM(is_churned) AS churned,
            ROUND(SUM(is_churned) * 100.0 / COUNT(*), 1) AS churn_rate_pct,
            SUM(CASE WHEN is_churned = 1 THEN arr_usd ELSE 0 END) AS churned_arr
        FROM accounts
        GROUP BY industry
        ORDER BY churn_rate_pct DESC
    """,

    "expansion_candidates": """
        -- Accounts with high usage growth and no open critical tickets
        -- (prime candidates for upsell conversation)
        WITH usage_trend AS (
            SELECT account_id,
                   MAX(api_calls) - MIN(api_calls) AS usage_growth
            FROM usage
            GROUP BY account_id
        )
        SELECT
            a.account_id,
            a.account_name,
            a.segment,
            a.arr_usd,
            a.health_score,
            ut.usage_growth
        FROM accounts a
        JOIN usage_trend ut ON a.account_id = ut.account_id
        WHERE a.is_churned = 0
          AND a.health_score > 70
          AND ut.usage_growth > 5000
        ORDER BY ut.usage_growth DESC
        LIMIT 10
    """,
}


def run_sql_demo(accounts_df, opportunities_df, usage_df):
    """Load data into SQLite and run raw SQL queries."""
    conn = sqlite3.connect(":memory:")
    accounts_df.to_sql("accounts", conn, index=False, if_exists="replace")
    opportunities_df.to_sql("opportunities", conn, index=False, if_exists="replace")
    usage_df.to_sql("usage", conn, index=False, if_exists="replace")

    print("\n" + "=" * 70)
    print("  RAW SQL QUERIES — SQLite Demo")
    print("=" * 70)
    for name, sql in RAW_SQL_QUERIES.items():
        print(f"\n--- {name} ---")
        result = pd.read_sql_query(sql, conn)
        print(result.to_string(index=False))

    conn.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RevOps Analytics Pipeline — Cohere Portfolio Demo")
    parser.add_argument("--export", action="store_true", help="Export all datasets to CSV")
    parser.add_argument("--sql",    action="store_true", help="Run raw SQL queries via SQLite")
    args = parser.parse_args()

    print("\n━" * 35)
    print("  REVOPS ANALYTICS PIPELINE")
    print("  Revenue Operations Analyst — Cohere Portfolio Demo")
    print("━" * 35)

    print("\nGenerating synthetic GTM data...")
    accounts      = generate_accounts(80)
    opportunities = generate_opportunities(accounts, 200)
    usage         = generate_usage(accounts)
    tickets       = generate_support_tickets(accounts)
    print(f"  {len(accounts)} accounts · {len(opportunities)} opportunities · "
          f"{len(usage)} usage rows · {len(tickets)} tickets")

    analytics = RevOpsAnalytics(accounts, opportunities, usage, tickets)
    analytics.print_report()

    if args.sql:
        run_sql_demo(accounts, opportunities, usage)

    if args.export:
        out = Path("revops_data")
        out.mkdir(exist_ok=True)
        accounts.to_csv(out / "accounts.csv", index=False)
        opportunities.to_csv(out / "opportunities.csv", index=False)
        usage.to_csv(out / "usage.csv", index=False)
        tickets.to_csv(out / "tickets.csv", index=False)
        analytics.churn_risk_model().to_csv(out / "churn_risk.csv", index=False)
        analytics.customer_360().to_csv(out / "customer_360.csv", index=False)
        print(f"\n  Exported 6 files to {out}/")


if __name__ == "__main__":
    main()
