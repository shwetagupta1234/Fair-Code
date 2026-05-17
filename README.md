# Fair Code — Algorithmic Bias Detection & Mitigation

> *AI systems are making decisions about your freedom, your job, and your healthcare. This project proves the bias is real — and shows exactly how to fix it.*

**by Yash Kewlani · [@thefaircodeproject](https://instagram.com/thefaircodeproject)**

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

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

---

## Projects

### 01 · COMPAS — Criminal Justice Bias

> *"A real algorithm used in US courtrooms flags Black defendants as high-risk at 87%. White defendants? 0.4%. Same system. Different outcomes."*

**Dataset:** `compas-scores-raw.csv` — ProPublica's public COMPAS dataset (70,000+ records)

COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) is used across 46 US states to predict whether a defendant will reoffend. Judges use its scores to make bail, sentencing, and parole decisions. More than 1 million people are assessed by COMPAS-style tools annually. Zero states require it to be audited for bias.

#### The Problem (`unfair.py`)

Trained with race and custody status as features — inputs COMPAS-style systems actually use in production.

| Group | High-Risk Flag Rate |
|---|---|
| Black Defendants | 87.16% |
| White Defendants | 0.40% |
| **Fairness Gap** | **86.77%** |

#### The Fix (`fair.py`)

Dropped race directly, and `CustodyStatus` as a known **proxy variable** — a correlated feature that smuggles racial signal back in even when the race column is removed.

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

> **Key Insight:** Removing race alone isn't enough. Proxy variables like custody status carry the same racial signal through the model because of historical over-policing of Black communities. Both the protected attribute and its proxies must be removed.

---

### 02 · AI Recruitment — Hiring Bias

> *"Women were hired 20.9% less than equally qualified men. The algorithm wasn't told to discriminate. It learned to."*

**Dataset:** `AI_Fair_Recruitment_Dataset.csv` — Recruitment dataset with gender, age, experience, and technical test scores

#### The Problem (`unfair.py`)

Biased model trained with gender and age alongside merit-based inputs:

| Group | Hire Rate |
|---|---|
| Men | 21.62% |
| Women | 17.10% |
| **Fairness Gap** | **4.51%** |

Women were hired **~21% less** than men with identical experience and test scores.

#### The Fix (`fair.py`)

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
├── explainers/
│   └── proxy-variables.md         # What is a proxy variable? (concept + detection code)
│
└── README.md
```

---

## Explainers

Technical concepts documented for anyone who wants to understand the "why" behind the code:

| Explainer | Concept |
|---|---|
| [What is a Proxy Variable?](explainers/proxy-variables.md) | Why AI stays biased even after you remove race from the data |
| Coming soon | What is demographic parity? |
| Coming soon | Why fairness metrics conflict with each other |

---

## Methodology

Both projects use the same bias detection and mitigation pipeline:

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

These aren't edge cases or hypotheticals. Algorithms like COMPAS are deployed in courtrooms today. Hiring AIs filter your resume before a human ever reads it. The bias in these systems is documented, measurable — and fixable.

---

## Getting Started

```bash
git clone https://github.com/yakew7/Fair-Code.git
cd Fair-Code
pip install pandas scikit-learn
```

**Run the COMPAS project:**
```bash
cd COMPAS
python unfair.py   # See the bias
python fair.py     # See the fix
```

**Run the recruitment project:**
```bash
cd "Ai Fair recrutment Dataset"
python unfair.py   # See the bias
python fair.py     # See the fix
```

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** `pandas`, `scikit-learn`
- **Datasets:** ProPublica COMPAS (public domain), AI Fair Recruitment Dataset (Kaggle)

---

## What's Next

- [ ] Facial recognition accuracy gaps (MIT Gender Shades methodology)
- [ ] HMDA mortgage lending bias
- [ ] Healthcare AI bias (Optum-style dataset)
- [ ] LLM bias audit
- [ ] Fairness audit web dashboard

---

## Connect

Follow the project on Instagram: **[@thefaircodeproject](https://instagram.com/thefaircodeproject)**  
Data. Code. Accountability. One post at a time.

---

*All datasets used are publicly available. This project is for educational and awareness purposes.*
