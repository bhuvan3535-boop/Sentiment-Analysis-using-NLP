# Sentiment-Analysis-using-NLP
This project is a full-stack data science application designed to perform real-time, granular emotion analysis on public discourse. Unlike traditional sentiment analysis that only detects positive/negative polarity, this system utilizes a custom-trained AOA-CNN to classify text into 28 distinct emotional categories 

The application integrates Google's Gemini LLM to provide qualitative context to the quantitative data, generating human-readable executive summaries of the prevailing discussions. The results are presented in an interactive Streamlit dashboard, offering both high-level metrics and deep-dives into raw data.

Developed a real-time AI dashboard capable of scraping and analyzing 500+ Reddit posts concurrently to detect granular public sentiment.

Engineered a hybrid intelligence pipeline combining a custom AOA-CNN (Attention-on-Attention) model for 28-class fine-grained emotion detection with Google Gemini LLM for generative thematic summarization.

Optimized system latency by ~65% by implementing ThreadPoolExecutor for parallel LLM API calls, enabling near-instant generation of Overall, Positive, and Negative thematic summaries.

Built an interactive Streamlit frontend featuring dynamic Plotly/Altair visualizations, enabling users to filter raw data and view real-time sentiment distribution.
Tech Stack & Tools
Core AI: TensorFlow/Keras (Custom CNN Model), Google Generative AI (Gemini Pro).
Data Acquisition: PRAW (Python Reddit API Wrapper).
Processing: NLTK (Natural Language Toolkit) for lemmatization and stopword removal, Pandas, NumPy.
Visualization & UI: Streamlit, Plotly Express (Donut Charts), Altair (Bar Charts).
Performance: Python concurrent.futures for multi-threading.
