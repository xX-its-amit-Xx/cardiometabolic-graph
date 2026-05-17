# Cookbook 02 — Re-engagement outreach targeting

## The question (from a DTx growth team)

> "We send re-engagement pushes to lapsing users. Most pushes get ignored.
> Can you give us the 200 lapsing users who are *most likely to respond* to
> the next push so we don't burn our send budget?"

This is dropout prediction inverted: instead of just flagging high dropout
risk, we want users who are **lapsing but recoverable** — high engagement in
the past, dropping now, but still showing micro-signals (a single recent
message-response, a stable glucose-log cadence) that suggest the lapse is
soft.

## What this example does

1. Loads the trained engagement-dropout classifier.
2. Scores every patient's dropout probability.
3. Defines "lapsing" as: `p_dropout > 0.40` AND `app_opens_30d > 0` AND
   `prior_30d_opens > 10`. (Pure abandonment without any prior usage is
   filtered out — sending pushes to never-active users is wasted budget.)
4. Within that pool, scores *recoverability* as
   `recent_responses + 0.5 * recent_glucose_logs - 0.1 * days_since_last_open`.
5. Ranks by recoverability and writes the top N to `outreach_targets.csv`.

## How a real team would use this

* CRM/email system consumes `outreach_targets.csv` weekly.
* Each row carries a *recovery score* you can A/B against — e.g., send to
  the top quartile only and use the bottom quartile as a control to measure
  uplift from the model itself, not from the message.

## Run

```bash
python -m cookbook.02_reengagement_outreach.run --top 200
```
