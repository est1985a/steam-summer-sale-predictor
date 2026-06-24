# 🎮 Steam Summer Sale Discount Predictor

A machine learning web app that predicts how deeply a Steam game is likely to be discounted during the annual Steam Summer Sale.

**🔗 Live app: [steam-summer-sale-predictor-bd4khuscctxchdjkgvhpqt.streamlit.app](https://steam-summer-sale-predictor-bd4khuscctxchdjkgvhpqt.streamlit.app/)**

---

## What it does

- Search for any Steam game by title
- Get a predicted Summer Sale discount percentage based on historical patterns
- See the last 5 years of actual Summer Sale history for that game
- Get a "Buy Now or Wait?" recommendation
- See the all-time historical low price on Steam

---

## How it works

### The data
The model was trained on **12,617 Steam games** with Summer Sale history, built by combining two data sources:

- **Kaggle Steam Games Dataset** — game metadata (genre, price, reviews, release date, publisher, achievements etc.)
- **IsThereAnyDeal API** — historical price and discount data, averaged across up to **10 years** of Summer Sales per game

### The model
A **Random Forest Regressor** trained to predict Summer Sale discount percentages. Predictions are rounded to the nearest 5% to reflect how Steam actually sets discounts.

**Model performance:**
- R² = 0.671 (explains ~67% of the variation in discount depth)
- MAE = 8.1% (predictions are typically within ±8 percentage points)

### Key features used
| Feature | Importance |
|---|---|
| Publisher discount history | 0.441 |
| Game age (years since release) | 0.214 |
| Review sentiment (% positive) | 0.043 |
| Number of reviews | 0.033 |
| Base price | 0.029 |
| Achievements count | 0.029 |
| Genre (Early Access, Action, Indie, Adventure) | varies |

The single strongest predictor is **publisher discount history** — publishers with a track record of deep discounts tend to keep discounting deeply, regardless of the individual game.

---

## Project journey

This was built as an independent project for the **GCI 2026 Summer** data science course at the University of Tokyo's Matsuo & Iwasawa Lab.

### Key findings along the way

**Data quality matters more than model choice**
During development, I discovered that the IsThereAnyDeal API history endpoint only returns 3 months of data by default. My original dataset was accidentally based on single recent sales rather than multi-year averages. Fixing this with the API's `since` parameter improved R² from 0.459 to 0.611.

**Publisher identity is the strongest signal**
Initially the model relied heavily on game age as a proxy for publisher behaviour. Once publisher historical discount rate was added as an explicit feature, it became the dominant predictor — confirming that *who makes the game* matters more than almost anything else.

**Random Forest outperformed XGBoost**
On this dataset (~12,000 games), Random Forest consistently outperformed XGBoost, likely due to the moderate dataset size where XGBoost's sequential boosting approach was more prone to overfitting.

**More data + better data > tuning**
The biggest single performance gains came from improving data quality and quantity, not hyperparameter tuning. The jump from the initial 1,335-game dataset to the final 12,617-game dataset (with correctly averaged multi-year targets) produced far larger improvements than any amount of model tuning.

---

## Model limitations

- Predictions are typically within ±8% but individual games can vary more
- Publishers with unusual strategies (e.g. Valve, FromSoftware, Wube Software/Factorio) are harder to predict accurately
- The model cannot predict games released after the training data was collected
- Tags (which carry some signal) cannot be retrieved from the Steam public API, so those features default to 0 for live predictions
- All prices shown in USD

---

## Tech stack

- **Python** — data collection, feature engineering, model training
- **scikit-learn** — Random Forest Regressor
- **Streamlit** — web app and deployment
- **Google Colab** — development environment
- **Steam Web API** — live game metadata
- **IsThereAnyDeal API** — historical price/discount data
- **Kaggle** — base dataset (Steam Games Dataset, March 2025)

---

## Questions or feedback?

Feel free to open an **Issue** on this repo — happy to discuss the methodology, answer questions about the model, or hear suggestions for improvement.

---

*Built as part of GCI 2026 Summer — University of Tokyo Matsuo & Iwasawa Lab*
