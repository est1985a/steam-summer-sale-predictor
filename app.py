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


def get_recent_summer_sales(appid, years_back=3):
    """Get actual Summer Sale discounts from the last N years, if available"""
    try:
        lookup_url = f"https://api.isthereanydeal.com/games/lookup/v1?key={ITAD_API_KEY}&appid={appid}"
        lookup_response = requests.get(lookup_url, timeout=10)
        lookup_data = lookup_response.json()

        if not lookup_data.get('found'):
            return []

        itad_id = lookup_data['game']['id']
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

        return sorted(yearly_discounts.items(), reverse=True)[:years_back]
    except Exception:
        return []


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
    price = price_data.get('final', 0) / 100
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
        'publisher': publisher,
        'game_age_years': round(game_age_years, 1),
        'header_image': metadata.get('header_image')
    }
    return result, None

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

                st.session_state.history = get_recent_summer_sales(
                    st.session_state.selected_appid
                )


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

        st.write(
            f"**Current price:** ${result['price']:.2f}"
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
            "This shows the deepest Steam discount found in June/July for each of the last 3 years, where available."
        )

    else:

        st.info(
            "No historical Summer Sale data found for this game — it may be new or rarely discounted."
        )


st.divider()

st.caption(
    "Built for GCI 2026 Summer — predictions based on a Random Forest model trained on 5,065 Steam games with multi-year Summer Sale history."
)
