import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import joblib

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Steam Summer Sale Predictor", page_icon="🎮", layout="centered")

st.title("🎮 Steam Summer Sale Discount Predictor")
st.write("Enter a Steam game title to predict its expected discount for the next Summer Sale.")

# ── Load model and data (cached so it only loads once) ─────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("steam_model.pkl")
    pub_freq_lookup = joblib.load("pub_freq_lookup.pkl")
    return model, pub_freq_lookup

@st.cache_data
def load_training_data():
    df = pd.read_csv("steam_summer_sale_dataset_v2.csv")
    return df

model, pub_freq_lookup = load_model()
df_steam = load_training_data()

ITAD_API_KEY = st.secrets["ITAD_API_KEY"]

FEATURES = [
    'game_age_years', 'pct_pos_total', 'log_publisher_frequency',
    'genre_early_access', 'log_achievements', 'log_price',
    'log_num_reviews_total', 'metacritic_score', 'log_peak_ccu',
    'log_average_playtime_forever', 'log_dlc_count',
    'genre_casual', 'genre_action', 'genre_rpg'
]

# ── Helper functions ─────────────────────────────────────────────────────────
def search_steam_game(game_name):
    url = "https://store.steampowered.com/api/storesearch"
    params = {'term': game_name, 'l': 'english', 'cc': 'US'}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if data['total'] == 0:
        return []
    return [{'appid': item['id'], 'name': item['name']} for item in data['items'][:5]]


def get_game_metadata(appid):
    url = "https://store.steampowered.com/api/appdetails"
    params = {'appids': appid, 'cc': 'us', 'l': 'english'}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if not data[str(appid)]['success']:
        return None
    return data[str(appid)]['data']


def get_review_data(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {'json': 1, 'language': 'all', 'purchase_type': 'all'}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    summary = data.get('query_summary', {})
    total_positive = summary.get('total_positive', 0)
    total_negative = summary.get('total_negative', 0)
    total = total_positive + total_negative
    if total == 0:
        return None
    return round((total_positive / total) * 100, 1)


def get_itad_id(appid):
    """Look up a game's internal ITAD id from its Steam appid"""
    lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&appid={appid}"
    lookup_response = requests.get(lookup_url, timeout=10)
    lookup_data = lookup_response.json()
    if not lookup_data.get('found'):
        return None
    return lookup_data['game']['id']


def get_steam_historical_low(itad_id):
    """Get the all-time historical low price specifically on Steam for this game"""
    try:
        prices_url = f"https://api.isthereanydeal.com/games/prices/v3?key={ITAD_API_KEY}&country=US"
        prices_response = requests.post(prices_url, json=[itad_id], timeout=10)
        prices_data = prices_response.json()

        if not prices_data:
            return None

        game_data = prices_data[0]

        for deal in game_data.get('deals', []):
            if deal['shop']['id'] == 61:  # Steam
                store_low = deal.get('storeLow', {})
                return store_low.get('amount')

        return None
    except Exception:
        return None


def get_recent_summer_sales(appid, years_back=5):
    """Get actual Summer Sale discounts from the last N years, plus all-time Steam low, if available"""
    try:
        itad_id = get_itad_id(appid)
        if itad_id is None:
            return [], None

        since_date = (datetime.datetime.now() - datetime.timedelta(days=365*years_back)).strftime('%Y-%m-%dT00:00:00Z')

        history_url = f"https://api.isthereanydeal.com/games/history/v2?key={ITAD_API_KEY}&id={itad_id}&country=US&since={since_date}"
        history_response = requests.get(history_url, timeout=10)
        history_data = history_response.json()

        yearly_discounts = {}
        for entry in history_data:
            if entry['shop']['id'] != 61:
                continue
            cut = entry['deal']['cut']
            if cut == 0:
                continue
            timestamp = entry['timestamp']
            year = int(timestamp[:4])
            month = int(timestamp[5:7])
            if month in [6, 7]:
                if year not in yearly_discounts or cut > yearly_discounts[year]:
                    yearly_discounts[year] = cut

        sale_history = sorted(yearly_discounts.items(), reverse=True)[:years_back]
        historical_low = get_steam_historical_low(itad_id)

        return sale_history, historical_low
    except Exception:
        return [], None


def predict_discount(appid):
    metadata = get_game_metadata(appid)
    if metadata is None:
        return None, "Game metadata not found on Steam."

    try:
        release_str = metadata['release_date']['date']
        release_date = pd.to_datetime(release_str)
        today = pd.Timestamp.today()
        game_age_years = (today - release_date).days / 365.25
    except Exception:
        return None, "Could not parse release date for this game."

    publisher = metadata.get('publishers', ['Unknown'])[0]
    pub_freq = pub_freq_lookup.get(publisher, 1)
    log_publisher_frequency = np.log1p(pub_freq)

    pct_pos_total = get_review_data(appid)
    if pct_pos_total is None:
        pct_pos_total = df_steam['pct_pos_total'].median()

    achievements_total = metadata.get('achievements', {}).get('total', 0)
    log_achievements = np.log1p(achievements_total)

    price_data = metadata.get('price_overview', {})
    price = price_data.get('initial', 0) / 100
    current_price = price_data.get('final', 0) / 100
    current_discount = price_data.get('discount_percent', 0) 
    log_price = np.log1p(price)

    num_reviews_total = metadata.get('recommendations', {}).get('total', 0)
    log_num_reviews_total = np.log1p(num_reviews_total)

    dlc_count = len(metadata.get('dlc', []))
    log_dlc_count = np.log1p(dlc_count)

    metacritic_score = metadata.get('metacritic', {}).get('score', 0)

    genres = [g['description'] for g in metadata.get('genres', [])]
    genre_early_access = 1 if 'Early Access' in genres else 0
    genre_casual = 1 if 'Casual' in genres else 0
    genre_action = 1 if 'Action' in genres else 0
    genre_rpg = 1 if 'RPG' in genres else 0

    log_peak_ccu = df_steam['log_peak_ccu'].median()
    log_average_playtime_forever = df_steam['log_average_playtime_forever'].median()

    features = pd.DataFrame([{
        'game_age_years': game_age_years,
        'pct_pos_total': pct_pos_total,
        'log_publisher_frequency': log_publisher_frequency,
        'genre_early_access': genre_early_access,
        'log_achievements': log_achievements,
        'log_price': log_price,
        'log_num_reviews_total': log_num_reviews_total,
        'metacritic_score': metacritic_score,
        'log_peak_ccu': log_peak_ccu,
        'log_average_playtime_forever': log_average_playtime_forever,
        'log_dlc_count': log_dlc_count,
        'genre_casual': genre_casual,
        'genre_action': genre_action,
        'genre_rpg': genre_rpg
    }])[FEATURES]

    prediction = model.predict(features)[0]

    result = {
        'name': metadata['name'],
        'predicted_discount': round(prediction, 1),
        'price': price,
        'current_price': current_price, 
        'current_discount': current_discount,
        'publisher': publisher,
        'game_age_years': round(game_age_years, 1),
        'header_image': metadata.get('header_image'),
        'appid': appid
    }
    return result, None

def buy_recommendation(predicted_discount, current_discount, price):
    """Generate a buy now or wait recommendation"""
    
    saving = price * (predicted_discount / 100)
    discounted_price = price * (1 - predicted_discount / 100)
    
    # Game is already on sale
    if current_discount > 0:
        current_saving = price * (current_discount / 100)
        return {
            'verdict': '🛒 Buy Now',
            'colour': 'green',
            'reason': f"This game is already {current_discount}% off (saving ${current_saving:.2f}). Don't wait — sale prices don't always come back immediately."
        }
    
    # Not on sale — predict future discount
    if predicted_discount < 20:
        return {
            'verdict': '🤷 Your call',
            'colour': 'orange',
            'reason': f"We predict only a small discount ({predicted_discount}%). Probably not worth waiting unless you're in no rush."
        }
    elif predicted_discount < 40:
        return {
            'verdict': '⏳ Consider Waiting',
            'colour': 'orange',
            'reason': f"A moderate discount of around {predicted_discount}% is likely, saving you around ${saving:.2f} (bringing it down to ~${discounted_price:.2f})."
        }
    else:
        return {
            'verdict': '⏰ Wait for the Sale',
            'colour': 'green',
            'reason': f"A significant discount of around {predicted_discount}% is predicted, saving you around ${saving:.2f} (bringing it down to ~${discounted_price:.2f})."
        }

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "selected_appid" not in st.session_state:
    st.session_state.selected_appid = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

if "history" not in st.session_state:
    st.session_state.history = None

if "historical_low" not in st.session_state:
    st.session_state.historical_low = None

# ── UI ───────────────────────────────────────────────────────────────────────
game_name = st.text_input(
    "Game title",
    placeholder="e.g. Hollow Knight"
)

if st.button("Predict discount", type="primary") and game_name:

    with st.spinner("Searching Steam..."):
        st.session_state.search_results = search_steam_game(game_name)

    st.session_state.prediction_result = None
    st.session_state.prediction_error = None
    st.session_state.history = None
    st.session_state.historical_low = None


if st.session_state.search_results:

    options = {
        f"{r['name']} (appid: {r['appid']})": r['appid']
        for r in st.session_state.search_results
    }

    choice = st.selectbox(
        "Select the exact game:",
        list(options.keys())
    )

    st.session_state.selected_appid = options[choice]

    if st.button("Confirm and predict"):

        with st.spinner("Fetching game data and predicting..."):

            result, error = predict_discount(
                st.session_state.selected_appid
            )

        st.session_state.prediction_error = error

        if not error:

            st.session_state.prediction_result = result

            with st.spinner("Checking historical sale data..."):

                history, historical_low = get_recent_summer_sales(
                    st.session_state.selected_appid,
                    years_back=5
                )

                st.session_state.history = history
                st.session_state.historical_low = historical_low


if st.session_state.prediction_error:

    st.error(st.session_state.prediction_error)


if st.session_state.prediction_result:

    result = st.session_state.prediction_result

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:

        if result["header_image"]:

            st.image(result["header_image"])

    with col2:

        st.subheader(result["name"])

        st.metric(
            "Predicted Summer Sale Discount",
            f"{result['predicted_discount']}%"
        )
        steam_url = f"https://store.steampowered.com/app/{result['appid']}/"
        st.markdown(f"[🔗 View on Steam]({steam_url})")
        
        # Buy recommendation
        rec = buy_recommendation(
        result['predicted_discount'],
        result['current_discount'],
        result['price']
        )

        st.markdown(f"### {rec['verdict']}")
        if rec['colour'] == 'orange':
            st.info(rec['reason'])
        else:
            st.success(rec['reason'])

        st.write(
            f"**Current price:** ${result['current_price']:.2f}"
        )

        st.write(
            f"**Publisher:** {result['publisher']}"
        )

        st.write(
            f"**Game age:** {result['game_age_years']} years"
        )

    st.divider()

    st.subheader("📊 Actual Summer Sale History")

    if st.session_state.history:

        hist_df = pd.DataFrame(
            st.session_state.history,
            columns=["Year", "Discount %"]
        )

        st.dataframe(
            hist_df,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "This shows the deepest Steam discount found in June/July for each of the last 5 years, where available."
        )

    else:

        st.info(
            "No historical Summer Sale data found for this game — it may be new or rarely discounted."
        )

    if st.session_state.historical_low is not None:

        st.metric(
            "All-time lowest price on Steam",
            f"${st.session_state.historical_low:.2f}"
        )

        st.caption(
            "The lowest price this game has ever been sold for on Steam, across any sale event (not just Summer Sale)."
        )


st.divider()

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
    ### About this tool
    This app uses a **Random Forest machine learning model** trained on data from 
    5,065 Steam games to predict how deeply a game might be discounted during the 
    Steam Summer Sale (typically held in late June / early July each year).
    
    ### How the prediction is made
    When you search for a game, the app fetches live data from Steam's API and 
    feeds it into the model. The model was trained on historical Summer Sale 
    discount data collected via the IsThereAnyDeal API, averaged across up to 
    5 years of Summer Sales per game.
    
    ### Features used to make the prediction
    The model considers the following signals:
    
    | Feature | Why it matters |
    |---|---|
    | **Game age** | Older games tend to be discounted more deeply |
    | **Publisher size** | Larger publishers have more predictable discount patterns |
    | **Review sentiment** | How positively reviewed the game is |
    | **Base price** | Higher priced games often have deeper % discounts |
    | **Number of reviews** | A proxy for overall popularity |
    | **Metacritic score** | Critical reception |
    | **Achievements** | A signal of game depth and engagement |
    | **DLC count** | Games with more DLC tend to discount the base game more |
    | **Genre** | Action, RPG, Casual, and Early Access games show different patterns |
    
    ### Model performance
    The model predicts Summer Sale discounts with a **mean absolute error of ~10%** 
    and explains around **61% of the variation** in discount depth across games (R² = 0.611).
    
    **What does that actually mean?**
    
    Imagine you asked 100 people to guess a game's Summer Sale discount just by looking 
    at the box — they'd probably be wildly off. Now imagine you gave them a detailed 
    history of every Steam sale ever, the game's review scores, its publisher's track 
    record, and how old the game is. They'd do much better.
    
    That's what this model does — it's learned patterns from 5,065 games and their 
    real Summer Sale history.
    
    - **Mean absolute error of ~10%** means that on average, if the model predicts 
    a 60% discount, the real discount typically falls somewhere between 50–70%. 
    Not perfect, but a useful ballpark.
    
    - **R² of 0.611** means the model can explain about 61% of why discounts vary 
    between games using the features above. The remaining 39% comes down to things 
    we can't measure — publisher business decisions, marketing budgets, internal 
    Valve negotiations, and plain randomness.
    
    Think of it like a weather forecast — it won't be exactly right every time, 
    but it's much more useful than just guessing.
    
    ### Prices
    All prices are shown in **USD** using Steam's US store pricing. 
    Steam uses regional pricing, so actual prices in your local currency may differ. 
    If you're in Japan or another region, the Steam page link will automatically 
    show your local price.
    
    ### What this tool can't predict
    - Publisher-specific business decisions or marketing strategies
    - Games releasing after the model's training data was collected
    - Whether Valve runs a Summer Sale in a given year (though they have every year since 2010)
    - Flash sales or publisher-specific promotions outside the Summer Sale window
    """)

st.caption(
    "Predictions based on a RandomForest model trained on 5,065 Steam games with multi-year Summer Sale history."
)
