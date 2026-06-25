# Customer Churn Prediction & Segmentation — End to End ML Project

---

## What is this project?

This is a complete End to End Machine Learning project built on a Telecom company dataset.
The project has two parts:

- **Part 1 — Churn Prediction:** Predict which customers are likely to leave the company (Supervised Learning)
- **Part 2 — Customer Segmentation:** Group customers into meaningful segments based on behaviour (Unsupervised Learning)

---

## What is Churn?

Churn means a customer leaving or stopping the use of a service.
In telecom, churn = customer cancelling their subscription.

Losing a customer is expensive — it costs more to acquire a new customer than to retain an existing one. Predicting churn early gives the business time to take action before the customer leaves.

---

## Dataset

- **Source:** Telco Customer Churn Dataset (Excel file)
- **Rows:** 7043 customers
- **Columns:** 33 features including demographics, services, charges, and churn label

| Column | Description |
|--------|-------------|
| Tenure Months | How long the customer has been with the company |
| Monthly Charges | How much they pay per month |
| Total Charges | Total amount paid so far |
| Contract | Month-to-month, One year, Two year |
| Internet Service | DSL, Fiber optic, No |
| Churn Value | 0 = Stayed, 1 = Churned (target variable) |

---

## Step by Step Process

### Step 1 — Exploratory Data Analysis (EDA)

Before building any model, the data was explored to find patterns:

- **Tenure vs Churn** — Customers who churn have lower tenure. They leave early.
- **Monthly Charges vs Churn** — Churners tend to pay higher monthly charges.
- **Contract vs Churn** — Month-to-month customers churn the most. Two year contract customers almost never churn.
- **Internet Service vs Churn** — Fiber optic customers churn more than DSL despite paying more.
- **Payment Method vs Churn** — Electronic check users churn significantly more than auto-pay users.

Why EDA? You cannot build a good model without understanding your data first. EDA helps find patterns, spot data problems, and decide which features are worth using.

---

### Step 2 — Data Cleaning

**Problems found and fixed:**
- `Total Charges` was stored as text (object type) instead of a number — converted using `pd.to_numeric(errors='coerce')`
- Blank values in `Total Charges` — filled with 0 using `fillna(0)`
- `City` column had 1500+ unique values — encoding it would create 1500+ new columns and add noise, so it was dropped

**Columns dropped and why:**

| Column | Reason |
|--------|--------|
| CustomerID, Count | Just identifiers, no pattern to learn |
| Country, State, City, Zip Code, Lat Long | Location data, not useful for churn prediction |
| Churn Label, Churn Score, CLTV, Churn Reason | Calculated AFTER churn happens — using these would be data leakage (cheating) |

Rule: Garbage in = Garbage out. If data has errors, the model learns wrong patterns.

---

### Step 3 — Feature Encoding

ML models only understand numbers. Columns like Contract and Payment Method had text values like "Month-to-month" or "Yes/No".

`pd.get_dummies(df, drop_first=True)` converts all text columns into 0/1 numeric columns.

Why `drop_first=True`? If a column has 3 categories, you only need 2 columns. If both are 0, the third category is automatically understood. Keeping all 3 creates redundancy which can confuse the model.

Result: 20 columns before encoding → 31 columns after encoding

---

### Step 4 — X and Y Split

```python
X = df_encoded.drop('Churn Value', axis=1)  # 30 features — everything we know about the customer
Y = df_encoded['Churn Value']                # target — will they churn? 0 or 1
```

X is the input. Y is what we want to predict.

---

### Step 5 — Train/Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Train: 5634 customers (80%)
# Test:  1409 customers (20%)
```

The model trains on 80% and is tested on the remaining 20% it has never seen. This measures how well the model generalises to new data rather than just memorising the training data.

`random_state=42` ensures the same split every run, making results reproducible.

---

## Part 1 — Churn Prediction Models

### Understanding Decision Trees and Random Forest

**What is a Decision Tree?**
A Decision Tree is a model that asks a series of yes/no questions about the data and reaches a prediction at the end. Think of it like a flowchart — "Is tenure less than 12 months? → Yes → Is monthly charge above $70? → Yes → Likely to churn."

**What is Depth in a Decision Tree?**
Depth controls how many levels of questions the tree can ask. A tree with depth=3 can ask 3 levels of questions. A tree with depth=20 can ask 20 levels.

**Problem with too little depth (underfitting):**
The tree cannot ask enough questions to understand the data properly. It oversimplifies and misses important patterns. Like a doctor who asks only 2 questions before diagnosing — not enough information.

**Problem with too much depth (overfitting):**
The tree asks too many questions and starts memorising the training data instead of learning general patterns. It performs well on training data but poorly on new data. Like a student who memorises answers instead of understanding concepts — fails on new questions.

**Sweet spot — depth=10:**
From our hyperparameter tuning, depth=10 gave the best balance of accuracy (78%) and recall (75%). Deep enough to learn real patterns, shallow enough to not memorise.

**What is Random Forest?**
Instead of one decision tree, Random Forest builds hundreds of trees — each trained on a slightly different random subset of data and features. All trees vote and the majority prediction wins. This removes the weakness of a single tree and gives much more stable, reliable predictions.

`n_estimators=100` means 100 trees are built. More trees = more stable but slower to train.

---

### Approach 1 — Basic Random Forest

```python
RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
```

```
Accuracy:     79%
Churn Recall: 50%   <- Problem: missing half the churners
```

**Problem found — Class Imbalance:**
The dataset has ~74% "Stay" customers and ~26% "Churn" customers. The model learned that predicting "Stay" most of the time gives decent accuracy, so it was biased towards "Stay." This caused it to miss 50% of actual churners — the exact customers the business needs to identify.

---

### Approach 2 — Balanced Random Forest [BEST MODEL]

```python
RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```

```
Accuracy:     79%
Churn Recall: 91%   <- massive improvement
AUC Score:    0.84
```

`class_weight='balanced'` tells the model to pay proportionally more attention to the minority class (churners). It automatically calculates weights so both classes are treated fairly during training.

**Why this is the best model:**
Recall is the most important metric for churn prediction, not accuracy. Missing a churner means the company loses that customer with no warning. A false alarm (predicting churn for someone who stays) only results in sending an unnecessary discount offer — a much cheaper mistake. Recall of 91% means the model catches 91 out of every 100 actual churners.

---

### Approach 3 — Feature Importance and Feature Selection

After training the balanced model, feature importance scores were extracted to understand which columns the model found most useful.

| Feature | Importance Score |
|---------|-----------------|
| Total Charges | 0.175 |
| Tenure Months | 0.170 |
| Monthly Charges | 0.148 |
| Contract Two year | 0.057 |
| Dependents Yes | 0.048 |

The top 3 features are all money and time related. Customers who pay more and stay longer are more predictable. Weak features `Phone Service_Yes` and `Multiple Lines_No phone service` were dropped since they added noise without contributing much.

---

### Approach 4 — Hyperparameter Tuning

Tried 20 combinations of n_estimators [100, 200, 300, 400, 500] and max_depth [5, 10, 15, 20]:

| Depth | Recall | Accuracy | Observation |
|-------|--------|----------|-------------|
| 5 | 82% | 74% | High recall but lower accuracy — trees too shallow, underfitting |
| 10 | 75% | 78% | Best balance — trees learned enough without memorising |
| 15 | 64% | 80% | Higher accuracy but missing more churners — too focused on majority class |
| 20 | 53% | 79% | Deep trees overfitting — memorising training data, poor generalisation |

**Key insight:** As depth increases beyond 10, accuracy goes up slightly but recall drops significantly. For this business problem, depth=10 is the sweet spot.

---

### Which Approach is Best and Why?

**Approach 2 — Balanced Random Forest is the winner.**

| Metric | Approach 1 Basic RF | Approach 2 Balanced RF | Approach 3 Feature Selected | Approach 4 Tuned RF |
|--------|--------------------|-----------------------|-----------------------------|---------------------|
| Accuracy | 79% | 79% | 78% | 78% |
| Churn Recall | 50% | **91%** | 74% | 75% |
| AUC Score | — | **0.84** | — | — |

Approach 2 wins because the business goal is to catch as many churners as possible. Recall of 91% vs 50% from the basic model is a massive improvement. Approaches 3 and 4 reduced recall by making the model simpler. The balanced approach with default depth gave the best real-world result for this problem.

---

### ROC AUC Curve

```
AUC Score: 0.84 — Very Good Model
```

AUC measures how well the model separates churners from non-churners across all decision thresholds.

| AUC Score | Meaning |
|-----------|---------|
| 0.5 | Random guessing — no better than a coin flip |
| 0.7 to 0.8 | Good model |
| 0.8 to 0.9 | Very good model — our score falls here |
| 0.9 and above | Excellent model |

Score of 0.84 means the model correctly identifies which customer is more likely to churn compared to another 84% of the time.

---

## Part 2 — Customer Segmentation (KMeans Clustering)

### What is Customer Segmentation?

Grouping customers into meaningful segments based on behaviour — without knowing the answer beforehand. This is Unsupervised Learning. There is no right or wrong answer — the algorithm finds natural groupings in the data.

Why segment? Different customers need different treatment. Sending the same offer to all customers wastes money. Targeted action based on segment is far more effective.

---

### Data Preparation

Four features were selected for segmentation:

- Tenure Months
- Monthly Charges
- Total Charges
- Churn Probability (from the Balanced Random Forest model)

Why include Churn Probability? This connects Part 1 and Part 2. Instead of clustering on raw data alone, the model's prediction is used as a feature — making the segments directly tied to churn risk, which is the business goal.

---

### Standard Scaling

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
```

KMeans works by measuring distances between data points. Total Charges goes up to 8000 but Tenure Months only goes to 72. Without scaling, Total Charges dominates the distance calculation simply because of its large range, not because it is actually more important. StandardScaler brings all features to the same scale (mean=0, standard deviation=1).

---

### Elbow Method — Finding Optimal K

Tried K from 1 to 15 and plotted WCSS (Within Cluster Sum of Squares) for each.

WCSS measures how compact the clusters are. Lower is better. But after a certain K, adding more clusters gives very little improvement — this point is called the elbow.

**Elbow found at K=3.** After K=3 the graph flattens significantly — adding more clusters is not worth the added complexity.

---

### KMeans with K=3 — Cluster Summary

| Cluster | Tenure | Monthly Charges | Churn Probability | Segment Name |
|---------|--------|-----------------|-------------------|--------------|
| 0 | 27 months | $38 | 7% | Budget Loyal Customers |
| 1 | 57 months | $90 | 13% | Loyal Premium Customers |
| 2 | 11 months | $74 | 71% | High Risk New Customers |

---

### Customer Segments Explained

**[Cluster 0] Budget Loyal Customers**
- Medium tenure (~27 months), low monthly charges (~$38), very low churn (7%)
- These customers have stayed for over 2 years but are on basic/cheap plans
- Business action: Personalised upsell campaigns — they trust the brand, conversion potential is high

**[Cluster 1] Loyal Premium Customers**
- High tenure (~57 months), high monthly charges (~$90), low churn (13%)
- The most valuable customers — been with the company for nearly 5 years, paying top prices
- Business action: Dedicated loyalty program, priority support, exclusive offers. Retention is critical.

**[Cluster 2] High Risk New Customers**
- Low tenure (~11 months), medium charges (~$74), very high churn probability (71%)
- New customers who are paying a decent amount but haven't built loyalty yet
- Business action: Immediate retention offers, better onboarding experience, priority support in first year

---

## Observations and Business Insights

### Observation 1 — New Customers Are the Biggest Risk
**Source:** Cluster Summary + Tenure Months vs Churn Probability scatter plot

Customers with tenure under 12 months have a churn probability of 71%. The scatter plot clearly shows that new customers (left side) are dominated by the High Risk segment. As tenure increases beyond 20 months, churn probability drops sharply.

**Business Insight:** The first year is the most critical retention window. Company should invest heavily in onboarding quality and offer incentives specifically for customers in their first 6 months.

---

### Observation 2 — High Charges Plus New Customer is the Danger Zone
**Source:** Tenure Months vs Monthly Charges scatter plot

High Risk New Customers cluster in the top left of the graph — low tenure but high monthly charges. These customers pay a lot but feel no loyalty because they are new.

**Business Insight:** Customers paying high amounts early but with no tenure feel the most pressure to leave if expectations are not met. Service guarantees and early-stage discounts for high-value new customers could significantly reduce this risk.

---

### Observation 3 — Contract Type is a Strong Churn Predictor
**Source:** EDA — Contract vs Churn countplot, Feature Importance

Month-to-month customers churn the most. Two year contract customers almost never churn. Contract type was also in the top 5 most important features for the model.

**Business Insight:** Long term contracts create loyalty. Incentivising customers to switch from monthly to annual contracts — through discounts or bundled services — would directly reduce churn rate.

---

### Observation 4 — Fiber Optic Customers Churn More Despite Paying More
**Source:** EDA — Internet Service vs Churn countplot

Fiber optic customers have higher churn than DSL customers despite paying a premium. This is counterintuitive — higher investment should mean more commitment.

**Business Insight:** The data suggests dissatisfaction with fiber optic service quality relative to price. This warrants a deeper investigation into fiber optic customer complaints and service reliability.

---

### Observation 5 — Electronic Check Payment Users Are High Risk
**Source:** EDA — Payment Method vs Churn countplot

Electronic check users churn significantly more than credit card, bank transfer, or auto-pay customers.

**Business Insight:** Electronic check is a manual payment — these customers are not locked in to auto-renewal. Encouraging a switch to auto-pay with a small monthly discount could passively reduce churn in this group.

---

### Observation 6 — The Model Has Direct Financial Impact
**Source:** Balanced Random Forest — 91% Churn Recall

The model correctly identifies 91 out of every 100 actual churners. Without this model, the company cannot proactively identify at-risk customers.

**Estimated impact:** With ~1800 customers churning per year at an average of $65/month, that is approximately $117,000 in monthly revenue loss from churned customers. If the model helps retain even 30% of predicted churners through targeted offers, it saves approximately $35,000 per month.

---

### Observation 7 — Budget Loyal Customers Are an Untapped Revenue Source
**Source:** Cluster 0 Summary

These customers have stayed 27 months on average but pay only $38/month — about half what premium customers pay. Their churn probability is just 7%, meaning they are satisfied and stable.

**Business Insight:** These customers represent a low-risk upsell opportunity. They are loyal enough to consider upgrades if offered in a personalised way. A targeted campaign for this group could increase average revenue per user without any churn risk.

---

### Summary of Business Actions

| Segment / Group | Problem | Recommended Action |
|-----------------|---------|-------------------|
| High Risk New Customers | 71% churn probability | Immediate retention offers, better onboarding |
| Loyal Premium Customers | Losing them is very costly | Loyalty rewards, priority customer support |
| Budget Loyal Customers | Paying too little for their loyalty | Personalised upgrade campaigns |
| Month-to-month contract users | Easy to leave with no commitment | Incentivise switch to annual contracts |
| Fiber optic customers | High churn despite high payments | Investigate and fix service quality issues |
| Electronic check users | Manual payment, less committed | Offer discount to switch to auto-pay |

---

## What I Learned — Skills Gained

### Machine Learning Concepts
- Supervised Learning — training a model when the answer (label) is known
- Unsupervised Learning — finding patterns when there is no known answer
- Classification — predicting a category (churn or not churn)
- Clustering — grouping data into natural segments without labels
- Class Imbalance — understanding why accuracy alone is misleading and how to fix it
- Overfitting vs Underfitting — understanding tree depth and its effect on generalisation
- Feature Importance — identifying which inputs the model relies on most
- Hyperparameter Tuning — systematically testing combinations to find optimal settings
- Model Evaluation — Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC AUC

### Data Skills
- Exploratory Data Analysis (EDA) — finding patterns before modelling
- Data Cleaning — handling missing values, wrong data types, irrelevant columns
- Feature Engineering — encoding categorical variables, scaling numerical features
- Train/Test Split — evaluating model on unseen data

### Tools and Technologies

| Tool | Purpose |
|------|---------|
| Python | Primary programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib | Plotting graphs and charts |
| Seaborn | Statistical visualisations |
| Scikit-learn | ML models — RandomForestClassifier, KMeans, StandardScaler, train_test_split, metrics |
| Google Colab | Cloud-based Jupyter notebook environment |

### Soft Skills
- Reading documentation and applying it practically
- Debugging errors independently (KeyError, NameError, wrong data types)
- Connecting model results to real business decisions
- Structuring a complete project end to end

---

## Final Results Summary

| Metric | Value |
|--------|-------|
| Best Model | Balanced Random Forest |
| Accuracy | 79% |
| Churn Recall | 91% |
| AUC Score | 0.84 |
| Customer Segments Found | 3 |
| High Risk Customers Churn Probability | 71% |
| Estimated Monthly Savings (if 30% retained) | ~$35,000 |

---

## About This Project

This project was built during my AI/ML internship at Coding Blocks. It covers the complete ML pipeline from raw data to actionable business insights — EDA, data cleaning, feature engineering, supervised classification, unsupervised clustering, and business recommendation generation.
