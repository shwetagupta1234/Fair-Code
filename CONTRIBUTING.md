# Contributing to Fair Code

Thanks for wanting to add to this. The goal is simple: find a real dataset where an AI system produces measurably biased outcomes, prove it with code, fix it, and prove that too.

Every contribution follows the same structure as the existing COMPAS and AI Recruitment audits. No exceptions — consistency is what makes the repo credible.

---

## Before You Start

Check the [roadmap in the README](README.md#whats-next) and open an Issue before doing any work. If two people are auditing the same dataset at the same time, that's wasted effort. The Issue is just a one-liner: "Taking on HMDA mortgage lending bias — starting with the federal HDMA dataset."

---

## Folder Structure

Every audit lives in its own top-level folder. Name it after the domain, not the dataset.

```
Fair-Code/
├── COMPAS/                        ← existing
├── Ai Fair recrutment Dataset/    ← existing
├── Your-Domain-Here/              ← your new audit
│   ├── unfair.py
│   ├── fair.py
│   ├── your-dataset.csv
│   ├── unfair.jpg
│   └── fair.jpg
```

No subfolders, no extra files. Keep it flat.

---

## The Two Scripts

Every audit has exactly two scripts. Nothing more.

### `unfair.py` — the biased model

This script demonstrates the bias. It must:

- Load the dataset
- Train a Random Forest Classifier with protected attributes included (race, gender, age, or whatever applies)
- Print results in this exact format:

```
--- BIASED MODEL RESULTS ---

[Group A] [Outcome] Rate: XX.XX%
[Group B] [Outcome] Rate: XX.XX%

Fairness Gap: XX.XX%
```

Use `random_state=42` and an 80/20 train/test split. Both are non-negotiable — they make results reproducible.

### `fair.py` — the mitigated model

This script fixes the bias. It must:

- Drop the protected attribute(s)
- Drop any identified proxy variables (see below)
- Retrain on the remaining features
- Print results in this exact format:

```
--- MITIGATED (UNBIASED) RESULTS ---

[Group A] [Outcome] Rate: XX.XX%
[Group B] [Outcome] Rate: XX.XX%

New Fairness Gap: XX.XX%
```

---

## Proxy Variables

This is the most important part of the audit. Removing the protected attribute alone is rarely enough.

A proxy variable is a feature that correlates with the protected attribute strongly enough to smuggle the bias back through the model — even after you've dropped race or gender directly. In the COMPAS audit, `CustodyStatus` was a proxy for race because of historical over-policing patterns.

You must identify and document your proxy variables. To find them:

```python
import pandas as pd

df = pd.read_csv('your-dataset.csv')

# Correlation between potential proxy and protected attribute
print(df[['potential_proxy_column', 'protected_attribute']].corr())

# Or for categorical variables, use a crosstab
print(pd.crosstab(df['potential_proxy'], df['protected_attribute'], normalize='index'))
```

If a feature shows strong correlation with the protected attribute and you include it in the fair model, explain why (i.e. it's a genuine merit signal, not a proxy). If you drop it, document it.

---

## The Output Screenshots

After running both scripts, take terminal screenshots and save them as:

- `unfair.jpg` — terminal output of `unfair.py`
- `fair.jpg` — terminal output of `fair.py`

These go in your audit folder alongside the scripts. They're the visual proof.

---

## Dataset Requirements

- Must be **publicly available** with a clear source (Kaggle, government data, ProPublica, academic release, etc.)
- Must contain **real demographic data** — synthetic datasets don't demonstrate real-world bias
- Include the dataset file in your folder if it's under ~50MB. If it's larger, add a `DATA.md` with a direct download link and instructions

Do not include datasets that require login, payment, or a data use agreement to access.

---

## Updating the README

Add your audit to the results table at the top of `README.md`:

```markdown
| [Your Domain](#link-to-section) | Bias Type | Gap Before | Gap After | Reduction |
```

Then add a full section below the existing projects following the same pattern: the quote, dataset description, the biased results table, the code snippet showing what you dropped, the mitigated results table, and the key insight paragraph.

The key insight paragraph is required. It should answer: *why did the bias exist, and why does the fix work?* One short paragraph, plain language.

---

## Fairness Metric

All audits use **Demographic Parity** as the primary metric: the difference in positive prediction rates between demographic groups.

If your domain genuinely requires a different metric (equalized odds, predictive parity, etc.), open an Issue to discuss it first before submitting.

---

## Submitting

1. Fork the repo
2. Create a branch: `git checkout -b audit/your-domain`
3. Add your folder with both scripts, the dataset, and both screenshots
4. Update `README.md`
5. Open a Pull Request with a title like `Audit: HMDA Mortgage Lending Bias`

In the PR description, include:
- What dataset you used and where it's from
- What the bias type is
- The before and after fairness gap numbers
- What proxy variables you found and why you dropped them

That's it.

---

## What Won't Be Merged

- Audits on synthetic or toy datasets
- Scripts that don't produce reproducible results (`random_state=42`, consistent split)
- Fair models that achieve parity by tanking overall accuracy to near-random
- Audits without identified proxy variables (unless you document why none exist)
- Any dataset that isn't publicly accessible without login or payment

---

*All datasets used in this project are publicly available. Fair Code is for educational and awareness purposes.*
