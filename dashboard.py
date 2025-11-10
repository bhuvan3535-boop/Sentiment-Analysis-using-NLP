import streamlit as st
import pandas as pd
import praw
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import plotly.express as px
import altair as alt
from collections import Counter
import re
import json
import warnings
import requests
import concurrent.futures
import time

# ----------------------------------------------------------------------
# PAGE CONFIG - MUST BE FIRST
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings and TF logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------------------------------------------------------------
# CUSTOM CSS FOR "INTERESTING" UI
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; white-space: pre-wrap; background-color: #f8f9fa;
        border-radius: 8px 8px 0px 0px; gap: 5px; padding: 10px 15px; font-weight: 600; border: 1px solid #e0e0e0; border-bottom: none; color: #31333F;
    }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; border-top: 3px solid #ff4b4b !important; }
    
    /* UPDATED: Added 'color: #000000' to force black text */
    .summary-box { padding: 20px; border-radius: 0px 10px 10px 10px; background-color: #ffffff; border: 1px solid #e0e0e0; font-size: 17px; line-height: 1.5; color: #000000; }
    
    .positive-box { border-left: 5px solid #00cc96; }
    .negative-box { border-left: 5px solid #EF553B; }
    .neutral-box { border-left: 5px solid #636efa; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# !!! 🔒 SECURE API KEYS HERE 🔒 !!!
# ----------------------------------------------------------------------
REDDIT_CLIENT_ID = "BKmcebeSU0uVKN_4MyDUdg"
REDDIT_CLIENT_SECRET = "BdxV0tR0eFKgfkqjWNs22qlrRXdERA"
REDDIT_USER_AGENT = "EmotionScraper v2.0 by /u/your_username"

GEMINI_API_KEY = "AIzaSyAPBf3im3qQwUsbhPhvll0HfE2dnb_61EY"
# ----------------------------------------------------------------------

MODEL_FILE = 'final_cnn_model.keras'
TOKENIZER_FILE = 'tokenizer.json'

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love',
    'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]
MAX_LEN = 100

AGGREGATE_MAP = {
    'admiration': 'positive', 'amusement': 'positive', 'approval': 'positive',
    'caring': 'positive', 'desire': 'positive', 'excitement': 'positive',
    'gratitude': 'positive', 'joy': 'positive', 'love': 'positive',
    'optimism': 'positive', 'pride': 'positive', 'relief': 'positive',
    'anger': 'negative', 'annoyance': 'negative', 'disappointment': 'negative',
    'disapproval': 'negative', 'disgust': 'negative', 'embarrassment': 'negative',
    'fear': 'negative', 'grief': 'negative', 'nervousness': 'negative',
    'remorse': 'negative', 'sadness': 'negative',
    'confusion': 'neutral', 'curiosity': 'neutral', 'realization': 'neutral',
    'surprise': 'neutral', 'neutral': 'neutral', 'N/A': 'neutral'
}

@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except: pass
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    try:
        with open(TOKENIZER_FILE, 'r', encoding='utf-8') as f: tokenizer = tokenizer_from_json(json.load(f))
        model = load_model(MODEL_FILE, compile=False)
    except Exception as e: return None, None, None, None, str(e)
    return lemmatizer, stop_words, tokenizer, model, None

@st.cache_resource
def get_reddit_client():
    if "YOUR_CLIENT_ID_HERE" in REDDIT_CLIENT_ID: return None
    return praw.Reddit(client_id=REDDIT_CLIENT_ID, client_secret=REDDIT_CLIENT_SECRET, user_agent=REDDIT_USER_AGENT)

lemmatizer, stop_words, tokenizer, model, load_error = load_resources()
reddit = get_reddit_client()
if load_error: st.error(f"CRITICAL ERROR: {load_error}"); st.stop()

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|@[^\s]+|[^a-zA-Z\s]', ' ', str(text).lower())
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2])

def get_prediction(raw_text_list, threshold=0.05):
    if not raw_text_list: return [], []
    seq = tokenizer.texts_to_sequences([clean_text(t) for t in raw_text_list])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0, batch_size=512)
    emotions, flat = [], []
    for p in probs:
        idx = np.where(p > threshold)[0]
        labels = [EMOTION_LABELS[i] for i in idx] if len(idx) > 0 else ["N/A"]
        emotions.append(labels)
        flat.extend(labels)
    return emotions, flat

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_reddit_posts(_reddit, keyword, max_posts, max_subs, posts_per_sub):
    if not _reddit: return pd.DataFrame()
    try: subs = [s.display_name for s in _reddit.subreddits.search(keyword, limit=max_subs)] or ['all']
    except: subs = ['all']
    posts = []
    for sub in subs:
        if len(posts) >= max_posts: break
        try:
            for s in _reddit.subreddit(sub).search(keyword, sort="relevance", limit=posts_per_sub):
                if len(posts) >= max_posts: break
                txt = (s.title or '') + ' ' + (s.selftext or '')
                if len(txt) > 50: posts.append({'Subreddit': f"r/{sub}", 'Text': txt, 'Score': s.score, 'URL': s.url})
        except: continue
    return pd.DataFrame(posts)

@st.cache_data(ttl=3600, show_spinner=False)
def get_gemini_summary_single(text_list, sentiment_type, topic_name):
    if not text_list: return f"No {sentiment_type} content found."
    if "YOUR_GEMINI_API_KEY_HERE" in GEMINI_API_KEY: return "MISSING API KEY"
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"
    prompt = f"Summarize key themes for {sentiment_type} sentiment on '{topic_name}'. STRICT LIMIT: under 100 words."
    if sentiment_type == "overall": prompt = f"Executive summary of discourse on '{topic_name}'. STRICT LIMIT: under 100 words."
    try:
        resp = requests.post(API_URL, headers={"Content-Type": "application/json"},
                             json={"contents": [{"parts": [{"text": f"Context: {'. '.join(text_list)[:25000]}\n\nTask: {prompt}"}]}]})
        resp.raise_for_status()
        return resp.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e: return f"Error: {str(e)[:50]}..."

# ----------------------------------------------------------------------
# MAIN APP UI
# ----------------------------------------------------------------------
st.title("🧠 Sentiment & Emotion AI Dashboard")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Analysis Mode", ["Reddit Topic", "Raw Text Input"])
    if mode == "Reddit Topic":
        with st.expander("Advanced Options"):
            max_p = st.slider("Max Posts", 100, 2000, 500, step=100)
            max_s = st.slider("Max Subreddits", 1, 10, 5)
            p_per_s = st.slider("Posts/Subreddit", 50, 500, 100)

if mode == "Reddit Topic":
    c_search, c_btn = st.columns([4, 1])
    with c_search: topic = st.text_input("Topic", placeholder="Enter topic...", label_visibility="collapsed")
    with c_btn: run = st.button("🚀 Launch", use_container_width=True, type="primary")

    if run and topic:
        if not reddit: st.error("⚠️ API Keys missing!"); st.stop()
        with st.status("💡 Processing...", expanded=True) as status:
            start_time = time.time()
            df = fetch_reddit_posts(reddit, topic, max_p, max_s, p_per_s)
            if df.empty: status.update(label="❌ No data!", state="error"); st.stop()
            emotions, flat_emotions = get_prediction(df['Text'].tolist(), threshold=0.1)
            df['emotions'] = emotions
            df['sentiment'] = [list(set([AGGREGATE_MAP.get(e, 'neutral') for e in el])) for el in emotions]
            with concurrent.futures.ThreadPoolExecutor() as ex:
                f_o = ex.submit(get_gemini_summary_single, df['Text'].tolist(), "overall", topic)
                f_p = ex.submit(get_gemini_summary_single, df[df['sentiment'].apply(lambda x: 'positive' in x)]['Text'].tolist(), "positive", topic)
                f_n = ex.submit(get_gemini_summary_single, df[df['sentiment'].apply(lambda x: 'negative' in x)]['Text'].tolist(), "negative", topic)
                summ_o, summ_p, summ_n = f_o.result(), f_p.result(), f_n.result()
            status.update(label="✅ Done!", state="complete", expanded=False)

        sent_counts = Counter([s for sub in df['sentiment'] for s in sub])
        
        # ROW 1: Donut (Left) | Metrics (Right 2x2)
        r1_c1, r1_c2 = st.columns([2, 3])
        with r1_c1:
            st.subheader("📊 Sentiment Split")
            fig = px.pie(pd.DataFrame(sent_counts.items(), columns=['S','C']), values='C', names='S', hole=0.5,
                         color='S', color_discrete_map={'positive':'#00cc96','negative':'#EF553B','neutral':'#636efa'})
            fig.update_traces(textposition='inside', textinfo='label+percent', textfont_size=16)
            fig.update_layout(showlegend=False, margin=dict(t=30,b=0,l=0,r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with r1_c2:
            st.subheader("⚡ At a Glance")
            m1, m2 = st.columns(2)
            m1.metric("📚 Posts", len(df), f"{time.time()-start_time:.1f}s")
            m1.metric("😃 Positive", sent_counts['positive'], "signals")
            m2.metric("😠 Negative", sent_counts['negative'], "signals", delta_color="inverse")
            m2.metric("😐 Neutral", sent_counts['neutral'], "signals", delta_color="off")

        # ROW 2: AI Summaries
        st.write("---")
        st.subheader("📝 AI Executive Briefing")
        t1, t2, t3 = st.tabs(["🌐 Overall", "📈 Positive", "📉 Negative"])
        t1.markdown(f'<div class="summary-box neutral-box">{summ_o}</div>', unsafe_allow_html=True)
        t2.markdown(f'<div class="summary-box positive-box">{summ_p}</div>', unsafe_allow_html=True)
        t3.markdown(f'<div class="summary-box negative-box">{summ_n}</div>', unsafe_allow_html=True)

        # ROW 3: Bar Chart & Data
        st.write("---")
        st.subheader("🌡️ Emotional Intensity")
        emo_df = pd.DataFrame(Counter(flat_emotions).items(), columns=['E','C']).sort_values('C', ascending=False).head(15)
        emo_df['T'] = emo_df['E'].map(AGGREGATE_MAP)
        st.altair_chart(alt.Chart(emo_df).mark_bar(cornerRadius=5).encode(
            x=alt.X('C', title='Mentions'), y=alt.Y('E', sort='-x', title=None),
            color=alt.Color('T', scale=alt.Scale(domain=['positive','negative','neutral'], range=['#00cc96','#EF553B','#636efa']), legend=None),
            tooltip=['E','C']).properties(height=400), use_container_width=True)
        
        with st.expander("🔎 Raw Data"):
            st.dataframe(df[['Subreddit','Score','Text','emotions','URL']], column_config={"URL": st.column_config.LinkColumn("Link")}, use_container_width=True)

elif mode == "Raw Text Input":
    st.subheader("📄 Text Analyzer")
    txt = st.text_area("Input text:", height=150)
    if st.button("Analyze", type="primary") and txt:
        with st.spinner("Analyzing..."):
            emos = get_prediction([txt], threshold=0.2)[0][0]
        if "N/A" in emos: st.info("No strong emotions.")
        else:
            st.write("### Signals:")
            c = st.columns(len(emos))
            for i, e in enumerate(emos):
                s = AGGREGATE_MAP.get(e,'neutral')
                col = "#00cc96" if s=='positive' else "#EF553B" if s=='negative' else "#636efa"
                c[i].markdown(f"<div style='text-align:center;padding:8px;background:{col}20;border-radius:8px;border:2px solid {col};color:{col};font-weight:bold'>{e.upper()}</div>", unsafe_allow_html=True)