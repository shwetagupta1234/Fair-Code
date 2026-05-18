# Fair Code — Algorithmic Bias Detection & Mitigation

> *AI systems are making decisions about your freedom, your job, and your healthcare. This project proves the bias is real — and shows exactly how to fix it.*

**by Yash Kewlani · [@thefaircodeproject](https://instagram.com/thefaircodeproject)**

---

## What This Is

Fair Code is an ongoing research and engineering project that exposes bias in real-world AI systems and demonstrates concrete mitigation strategies. Every project follows the same structure:

**train a biased model → measure the fairness gap → engineer a fair model → measure again**

No theory. Just data, code, and results.

---

## Results at a Glance

| Project | Bias Type | Gap Before | Gap After | Reduction |
|---|---|---|---|---|
| [COMPAS](#01--compas--criminal-justice-bias) | Racial | 86.77% | 15.69% | **71%** |
| [AI Recruitment](#02--ai-recruitment--hiring-bias) | Gender | 4.51% | 0.12% | **97.3%** |
| [German Credit Lending](#03--german-credit--lending-bias) | Age | 7.16% | 1.89% | **73.6%** |

---

## Projects

### 01 · COMPAS — Criminal Justice Bias

> *"A real algorithm used in US courtrooms flags Black defendants as high-risk at 87%. White defendants? 0.4%. Same system. Different outcomes."*

**Dataset:** `compas-scores-raw.csv` — ProPublica's public COMPAS dataset (70,000+ records)

COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is used across 46 US states to predict whether a defendant will reoffend. Judges use its scores to make bail, sentencing, and parole decisions. More than 1 million people are assessed by COMPAS-style tools annually. Zero states require it to be audited for bias.

**The Problem (`unfair.py`)**

Trained with race and custody status as features — inputs COMPAS-style systems actually use in production.

| Group | High-Risk Flag Rate |
|---|---|
| Black Defendants | 87.16% |
| White Defendants | 0.40% |
| **Fairness Gap** | **86.77%** |

**The Fix (`fair.py`)**

Dropped race directly, and CustodyStatus as a known proxy variable — a correlated feature that smuggles racial signal back in even when the race column is removed.

```python
# THE FIX: Drop race + proxy variables
X = pd.get_dummies(df[[
    'Sex_Code_Text',
    'MaritalStatus'
    # Race removed ✓
    # CustodyStatus removed ✓ (proxy for race due to over-policing)
]])
```

| Group | High-Risk Flag Rate |
|---|---|
| Black Defendants | 84.71% |
| White Defendants | 69.02% |
| **New Fairness Gap** | **15.69%** |

**Result: 71% reduction in the fairness gap.**

**Key Insight:** Removing race alone isn't enough. Proxy variables like custody status carry the same racial signal through the model because of historical over-policing of Black communities. Both the protected attribute and its proxies must be removed.

---

### 02 · AI Recruitment — Hiring Bias

> *"Women were hired 20.9% less than equally qualified men. The algorithm wasn't told to discriminate. It learned to."*

**Dataset:** `AI_Fair_Recruitment_Dataset.csv` — Recruitment dataset with gender, age, experience, and technical test scores

**The Problem (`unfair.py`)**

Biased model trained with gender and age alongside merit-based inputs:

| Group | Hire Rate |
|---|---|
| Men | 21.62% |
| Women | 17.10% |
| **Fairness Gap** | **4.51%** |

Women were hired ~21% less than men with identical experience and test scores.

**The Fix (`fair.py`)**

Dropped gender and age entirely. Retained only merit-based features: experience years and technical test score.

```python
# THE FIX: Merit only
X = df[['experience_years', 'test_score']]
# gender removed ✓
# age removed ✓
```

| Group | Hire Rate |
|---|---|
| Men | 11.48% |
| Women | 11.35% |
| **New Fairness Gap** | **0.12%** |

**Result: 97.3% reduction in the fairness gap.**

---

### 03 · German Credit — Lending Bias

> *"A credit scoring model rates young applicants as bad credit risks at 6.39 percentage points higher than older applicants with identical financial profiles. The algorithm doesn't know it's discriminating. It learned age from job tenure."*

**Dataset:** `credit_customers.csv` — German Credit Risk dataset, UCI Statlog (1,000 records). [Kaggle source](https://www.kaggle.com/datasets/ppb00x/credit-risk-customers) — public domain, no login required.

Age discrimination in lending is documented across financial systems worldwide. Young borrowers face higher rejection rates and worse terms not because of creditworthiness, but because the features used to measure creditworthiness — employment tenure, account history, savings — are structurally correlated with age. A 25-year-old with a stable income and no debt is not a worse credit risk than a 45-year-old with the same profile. But a model trained on historical data will treat them differently.

**The Problem (`unfair.py`)**

Biased model trained with `age` and `employment` (tenure) included as features:

| Group | Good Credit Rate |
|---|---|
| Older Applicants (30+) | 83.97% |
| Young Applicants (<30) | 76.81% |
| **Fairness Gap** | **7.16%** |

**Proxy Variable: `employment` (tenure)**

```python
import pandas as pd
df = pd.read_csv('credit_customers.csv')
df['is_young'] = (df['age'] < 30).astype(int)

print(pd.crosstab(df['employment'], df['is_young'], normalize='columns').round(3))

# Result:
# is_young          0      1
# employment
# <1yr           0.113  0.272   ← young applicants have short tenure at 2.4x the rate
# >=7yr          0.359  0.073   ← older applicants have long tenure at 4.9x the rate
```

Employment tenure is not an independent signal — it is structurally determined by age. A 24-year-old cannot have 10 years of employment history. When the model learns that short tenure predicts default, it is partially learning that being young predicts default. The feature smuggles age back through even after the `age` column is dropped.

**The Fix (`fair.py`)**

Dropped `age` and `employment`. Retained only objective financial signals that a borrower can control regardless of how old they are.

```python
# THE FIX: Financial signals only
features = [
    'checking_status',
    'duration',
    'credit_history',
    'purpose',
    'credit_amount',
    'savings_status',
    # employment removed ✓ (proxy: tenure is a direct function of age)
    'installment_commitment',
    'personal_status',
    'other_parties',
    'residence_since',
    'property_magnitude',
    # age removed ✓ (protected attribute)
    'other_payment_plans',
    'housing',
    'existing_credits',
    'job',
    'num_dependents',
    'own_telephone',
    'foreign_worker',
]
```

| Group | Good Credit Rate |
|---|---|
| Older Applicants (30+) | 80.15% |
| Young Applicants (<30) | 78.26% |
| **New Fairness Gap** | **1.89%** |

**Result: 73.6% reduction in the fairness gap.**

**Key Insight:** The bias in credit scoring isn't always intentional — it's structural. Employment tenure looks like a legitimate financial signal, and in isolation it is. But it's also a near-perfect proxy for age. A model that penalizes short tenure is partially penalizing youth, regardless of whether the word "age" appears anywhere in the feature list. Removing both age and its proxy forces the model to evaluate what a borrower has — their account balance, credit history, loan purpose, savings — rather than how long they've been alive.

---

## Repository Structure

```
Fair-Code/
│
├── COMPAS/
│   ├── unfair.py                  # Biased model (race + custody status included)
│   ├── fair.py                    # Mitigated model (race + proxy removed)
│   ├── compas-scores-raw.csv      # ProPublica COMPAS dataset
│   ├── unfair.jpg                 # Terminal output — biased results
│   └── fair.jpg                   # Terminal output — mitigated results
│
├── Ai Fair recrutment Dataset/
│   ├── unfair.py                  # Biased model (gender + age included)
│   ├── fair.py                    # Mitigated model (merit only)
│   ├── AI_Fair_Recruitment_Dataset.csv
│   ├── unfair.jpg                 # Terminal output — biased results
│   └── fair.jpg                   # Terminal output — mitigated results
│
├── German Credit Lending/
│   ├── unfair.py                  # Biased model (age + employment tenure included)
│   ├── fair.py                    # Mitigated model (financial signals only)
│   ├── credit_customers.csv       # UCI German Credit dataset
│   ├── unfair.jpg                 # Terminal output — biased results
│   └── fair.jpg                   # Terminal output — mitigated results
│
├── explainers/
│   └── proxy-variables.md         # What is a proxy variable? (concept + detection code)
│
└── README.md
```

---

## Explainers

| Explainer | Concept |
|---|---|
| [What is a Proxy Variable?](explainers/proxy-variables.md) | Why AI stays biased even after you remove race from the data |
| Coming soon | What is demographic parity? |
| Coming soon | Why fairness metrics conflict with each other |

---

## Methodology

All projects use the same bias detection and mitigation pipeline:

```
1. Load dataset
2. Train biased model (protected attributes included)
3. Measure fairness gap across demographic groups
4. Identify proxy variables via correlation analysis
5. Remove protected attributes + known proxy variables
6. Retrain fair model (merit features only)
7. Measure fairness gap again
8. Compare
```

**Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)  
**Split:** 80/20 train/test, `random_state=42`  
**Fairness Metric:** Demographic Parity — difference in positive prediction rates across groups  
**Mitigation Strategy:** Pre-processing attribute dropping (protected attributes + proxy variables)

---

## Why This Matters

- **87%** of companies use AI to screen job applicants before a human sees a resume
- **46** US states have used algorithmic risk tools in criminal sentencing
- **1M+** people are assessed by COMPAS-style tools annually
- **0** states require the algorithm to be audited for bias

These aren't edge cases or hypotheticals. Algorithms like COMPAS are deployed in courtrooms today. Hiring AIs filter your resume before a human ever reads it. Credit scoring models penalize young borrowers for not having lived long enough to build tenure. The bias in these systems is documented, measurable — and fixable.

---

## Getting Started

```
git clone https://github.com/yakew7/Fair-Code.git
cd Fair-Code
pip install pandas scikit-learn
```

**Run the COMPAS project:**
```
cd COMPAS
python unfair.py   # See the bias
python fair.py     # See the fix
```

**Run the recruitment project:**
```
cd "Ai Fair recrutment Dataset"
python unfair.py   # See the bias
python fair.py     # See the fix
```

**Run the German Credit project:**
```
cd "German Credit Lending"
python unfair.py   # See the bias
python fair.py     # See the fix
```

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `scikit-learn`
- **Datasets:** ProPublica COMPAS (public domain), AI Fair Recruitment Dataset (Kaggle), UCI German Credit / Statlog (Kaggle)

---

## What's Next

- [ ] Facial recognition accuracy gaps (MIT Gender Shades methodology)
- [ ] HMDA mortgage lending bias
- [ ] LLM bias audit
- [ ] Fairness audit web dashboard

---

## Connect

Follow the project on Instagram: **[@thefaircodeproject](https://instagram.com/thefaircodeproject)**  
Data. Code. Accountability. One post at a time.

---

*All datasets used are publicly available. This project is for educational and awareness purposes.*
