import streamlit as st
import os
import sqlite3
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.ndimage import maximum_filter
import static_ffmpeg
import pandas as pd
import tempfile
import time
import urllib.request

# Page configuration
st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for dark-premium theme matching the demo video
st.markdown("""
<style>
    /* Main layout modifications */
    .reportview-container {
        background-color: #0d0e12;
    }
    
    /* Title and Subtitle styling */
    .title-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .soundwave-icon {
        background-color: #172d2f;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #2ebd9a;
        color: #2ebd9a;
        font-size: 24px;
        font-weight: bold;
    }
    .main-title {
        color: #ffffff;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }
    .subtitle {
        color: #8e9ca3;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }
    .description-text {
        color: #b0c0c7;
        font-size: 14px;
        margin-bottom: 25px;
    }
    
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        border-bottom: 1px solid #2e2e38 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding-top: 5px;
        background-color: transparent !important;
        border: none !important;
        font-weight: bold !important;
        color: #8e9ca3 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #2ebd9a !important;
        border-bottom: 2px solid #2ebd9a !important;
    }
    
    /* Custom Info Box */
    .custom-info-box {
        background-color: #111216;
        border: 1px dashed #2d2e38;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        color: #8e9ca3;
        font-family: monospace;
        font-size: 14px;
        margin-bottom: 30px;
    }
    
    /* Database Card Styling */
    .db-card {
        background-color: #121318;
        border: 1px solid #1e2029;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .db-card-title {
        color: #ffffff;
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .db-card-subtitle {
        color: #8e9ca3;
        font-size: 12px;
        margin-bottom: 10px;
    }
    
    /* Performance metrics box styling */
    .metrics-container {
        background-color: #121318;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #1e2029;
        margin-bottom: 25px;
    }
    .metrics-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .metrics-title {
        font-size: 11px;
        color: #8e9ca3;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .metrics-total {
        font-size: 14px;
        color: #2ebd9a;
        font-weight: bold;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
    }
    .metric-item {
        text-align: center;
        border-right: 1px solid #1e2029;
    }
    .metric-item:last-child {
        border-right: none;
    }
    .metric-label {
        font-size: 9px;
        color: #8e9ca3;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 18px;
        color: #ffffff;
        font-weight: bold;
        margin-top: 3px;
    }
    .metric-sub {
        font-size: 10px;
        color: #2ebd9a;
        margin-top: 1px;
    }
    
    /* Match Found Container */
    .match-container {
        background-color: #101917;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2ebd9a;
        margin-bottom: 25px;
    }
    .match-label {
        font-size: 10px;
        color: #2ebd9a;
        font-weight: bold;
        letter-spacing: 1.5px;
    }
    .match-title {
        font-size: 32px;
        color: #ffffff;
        font-weight: 800;
        margin-top: 5px;
        margin-bottom: 5px;
    }
    .match-details {
        font-size: 13px;
        color: #8e9ca3;
    }
    .match-accent {
        color: #e5933a;
        font-weight: bold;
    }
    
    /* Candidate scores table-style grid */
    .scores-container {
        background-color: #121318;
        border: 1px solid #1e2029;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 25px;
    }
    .scores-title {
        font-size: 11px;
        color: #8e9ca3;
        font-weight: bold;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }
    .score-row {
        display: grid;
        grid-template-columns: 220px 1fr 60px;
        align-items: center;
        gap: 15px;
        margin-bottom: 12px;
    }
    .score-name {
        font-size: 13px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .score-bar-bg {
        background-color: #1b1c24;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
    }
    .score-bar-fill {
        background-color: #2ebd9a;
        height: 100%;
        border-radius: 3px;
    }
    .score-value {
        font-size: 13px;
        color: #ffffff;
        text-align: right;
        font-weight: bold;
    }
    
    /* Section Headers for Steps */
    .step-section {
        border-left: 3px solid #2ebd9a;
        padding-left: 15px;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .step-num {
        font-size: 10px;
        color: #2ebd9a;
        font-weight: bold;
        letter-spacing: 1px;
    }
    .step-title {
        font-size: 18px;
        color: #ffffff;
        font-weight: bold;
        margin-top: 2px;
    }
    .step-desc {
        color: #b0c0c7;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Custom header
st.markdown("""
<div class="title-container">
    <div class="soundwave-icon">🎛️</div>
    <div class="main-title">EE200: Audio Fingerprinting</div>
</div>
<div class="subtitle">Signals, Systems & Networks - Project Demo</div>
<div class="description-text">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
""", unsafe_allow_html=True)

# Initialize FFmpeg path
@st.cache_resource
def init_ffmpeg():
    try:
        static_ffmpeg.add_paths()
    except Exception as e:
        # On cloud Linux servers where packages.txt installs ffmpeg system-wide,
        # static_ffmpeg is not needed and might fail due to write permission locks.
        pass

init_ffmpeg()

# Database File Paths
DB_SQLITE = "fingerprint_db.sqlite"

# ----------------- CONFIGURATION FOR LIVE LINK DEPLOYMENT -----------------
# If deploying to GitHub/Streamlit Cloud, you can upload fingerprint_db.sqlite
# as a release asset in a GitHub release, and configure your username/repo below.
# The app will automatically download it on startup!
GITHUB_USERNAME = "moduguakhileshkumar"  # Replace with your GitHub Username
REPO_NAME = "Audio-Fingerprinting"              # Replace with your GitHub Repo Name
RELEASE_TAG = "V1.0"                      # Replace with your Release Tag

def check_and_download_db():
    if os.path.exists(DB_SQLITE):
        return True
    
    # Try downloading from GitHub Release if configured
    if GITHUB_USERNAME != "YOUR_GITHUB_USERNAME" and REPO_NAME != "YOUR_REPO_NAME":
        url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{DB_SQLITE}"
        try:
            with st.spinner("📥 Downloading SQLite database from GitHub Releases (373MB)... This will take a moment."):
                urllib.request.urlretrieve(url, DB_SQLITE)
            return True
        except Exception as e:
            st.error(f"Failed to download database from GitHub: {e}")
            return False
    return False

db_available = check_and_download_db()

if not os.path.exists(DB_SQLITE):
    st.error("""
    ### 📂 SQLite Database File Missing!
    To run this application, **`fingerprint_db.sqlite`** must be in the app directory.
    
    * **Running Locally**: Make sure you run `convert_to_sqlite.py` first to generate the SQLite database from your pickle file.
    * **Deploying to the Cloud**: Create a GitHub Release in your repository, upload `fingerprint_db.sqlite` as a release asset, and set `GITHUB_USERNAME` and `REPO_NAME` in `app.py` to enable automatic download.
    """)
    st.stop()

# Helper function to get all song names from SQLite
@st.cache_resource
def get_song_names():
    conn = sqlite3.connect(DB_SQLITE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM song_names ORDER BY song_idx")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names

song_names = get_song_names()

# Load only the 8 specific visible songs info for Tab 1 (LIBRARY) to keep startup instant
@st.cache_resource
def get_library_info(song_names):
    conn = sqlite3.connect(DB_SQLITE)
    cursor = conn.cursor()
    
    target_visible_songs = [
        "The Long And Winding Road",
        "Two Of Us",
        "Within You Without You",
        "A Hard Day's Night",
        "Let It Be",
        "Lucy In The Sky With Diamonds",
        "Penny Lane",
        "We Can Work It Out"
    ]
    
    library_info = {}
    for name in target_visible_songs:
        try:
            idx = song_names.index(name)
            
            # Query hash count
            cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE song_idx=?", (idx,))
            hash_count = cursor.fetchone()[0]
            
            # Query a sample of 1200 peaks to draw the tiny plot (t_db is x, decode f1 from hash_int)
            cursor.execute("SELECT t_db, (hash_int >> 16) FROM fingerprints WHERE song_idx=? LIMIT 2000", (idx,))
            peaks = list(set(cursor.fetchall()))
            peaks.sort(key=lambda x: x[0])
            
            library_info[idx] = {
                "name": name,
                "hashes_count": hash_count,
                "peaks": peaks
            }
        except ValueError:
            pass # Song not in database
            
    conn.close()
    return library_info

library_info = get_library_info(song_names)

# Helper functions for processing and matching
def get_peaks(y, sr):
    # Compute spectrogram
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Use maximum filter to find local peaks
    local_max = (S_db == maximum_filter(S_db, size=(15, 15)))
    background = (S_db > -45) # Threshold to discard quiet parts
    peaks = local_max & background
    
    freq_idxs, time_idxs = np.where(peaks)
    return list(zip(freq_idxs, time_idxs)), S, S_db

def generate_hashes(peaks):
    hashes = []
    peaks = sorted(peaks, key=lambda x: x[1])
    n_peaks = len(peaks)
    
    min_delta = 1
    max_delta = 30
    for i in range(n_peaks):
        f1, t1 = peaks[i]
        for j in range(i + 1, n_peaks):
            f2, t2 = peaks[j]
            dt = t2 - t1
            if min_delta <= dt <= max_delta:
                hashes.append(((f1, f2, dt), t1))
            elif dt > max_delta:
                break
    return hashes

# SQLite batch matching algorithm (extremely low memory usage)
def match_query(q_hashes):
    conn = sqlite3.connect(DB_SQLITE)
    cursor = conn.cursor()
    
    match_counts = defaultdict(int)
    song_offsets = defaultdict(list)
    
    # Map hash_int -> t_q
    hash_to_tq = {}
    for h_key, t_q in q_hashes:
        f1, f2, dt = h_key
        hash_int = int((f1 << 16) | (f2 << 8) | dt)
        hash_to_tq[hash_int] = t_q
        
    hash_ints = list(hash_to_tq.keys())
    
    # Query database in chunks of 500 hashes
    chunk_size = 500
    for i in range(0, len(hash_ints), chunk_size):
        chunk = hash_ints[i:i+chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        cursor.execute(f"SELECT hash_int, song_idx, t_db FROM fingerprints WHERE hash_int IN ({placeholders})", chunk)
        
        for hash_int, song_idx, t_db in cursor.fetchall():
            t_q = hash_to_tq[hash_int]
            offset = t_db - t_q
            match_counts[(song_idx, offset)] += 1
            song_offsets[song_idx].append(offset)
            
    conn.close()
    
    song_scores = defaultdict(int)
    best_offsets = {}
    for (song_idx, offset), count in match_counts.items():
        if count > song_scores[song_idx]:
            song_scores[song_idx] = count
            best_offsets[song_idx] = offset
            
    return song_scores, best_offsets, song_offsets


# Render Tabs
tab_lib, tab_ident, tab_batch = st.tabs(["❖ LIBRARY", "❂ IDENTIFY", "▩ BATCH"])

# ==========================================
# TAB 1: LIBRARY
# ==========================================
with tab_lib:
    st.markdown('<div class="custom-info-box">Song indexing is managed by the admin.<br>Drop a clip in the Identify tab to test the library.</div>', unsafe_allow_html=True)
    st.write("#### IN THE DATABASE")
    
    target_visible_songs = [
        "The Long And Winding Road",
        "Two Of Us",
        "Within You Without You",
        "A Hard Day's Night",
        "Let It Be",
        "Lucy In The Sky With Diamonds",
        "Penny Lane",
        "We Can Work It Out"
    ]
    
    # Colors for the constellation card scatter plots to match screenshots
    card_colors = [
        "#2ebd9a", # teal
        "#e5933a", # orange/yellow
        "#8884d8", # purple/indigo
        "#ff4d6d", # pink/red
        "#82ca9d", # green
        "#2ebd9a", # teal
        "#e5933a", # orange/yellow
        "#82ca9d"  # green
    ]
    
    cols = st.columns(4)
    card_idx = 0
    
    # Draw cards for the 8 visible songs
    for idx, song_info in library_info.items():
        col = cols[card_idx % 4]
        
        with col:
            # Container card
            st.markdown(f"""
            <div class="db-card">
                <div class="db-card-title">{song_info['name']}</div>
                <div class="db-card-subtitle">{song_info['hashes_count']:,} hashes</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sub-plot constellation visualization (fast render using matplotlib Agg)
            fig, ax = plt.subplots(figsize=(3.5, 1.6))
            fig.patch.set_facecolor('#121318')
            ax.set_facecolor('#121318')
            
            peaks = song_info['peaks']
            step = max(1, len(peaks) // 400)
            sampled_peaks = peaks[::step]
            
            times = [p[0] for p in sampled_peaks]
            freqs = [p[1] for p in sampled_peaks]
            
            ax.scatter(times, freqs, color=card_colors[card_idx % len(card_colors)], s=1.5, alpha=0.6)
            ax.set_xlim(0, max(times) if times else 6000)
            ax.set_ylim(0, 512)
            ax.axis('off')
            
            st.pyplot(fig)
            st.write("") # Spacer
            card_idx += 1

    # Toggle for other songs in the database
    with st.expander("Show other songs in database"):
        conn = sqlite3.connect(DB_SQLITE)
        cursor = conn.cursor()
        cursor.execute("SELECT song_idx, name FROM song_names")
        all_songs = cursor.fetchall()
        
        other_songs_list = []
        for s_idx, name in all_songs:
            if name not in target_visible_songs:
                # Query hash count
                cursor.execute("SELECT COUNT(*) FROM fingerprints WHERE song_idx=?", (s_idx,))
                hash_count = cursor.fetchone()[0]
                other_songs_list.append({
                    "Song Index": s_idx + 1,
                    "Song Title": name,
                    "Total Hashes": f"{hash_count:,}"
                })
        conn.close()
        st.dataframe(pd.DataFrame(other_songs_list), use_container_width=True)

# ==========================================
# TAB 2: IDENTIFY
# ==========================================
with tab_ident:
    st.write("### Identify a clip")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed")
    
    st.write("##### OR TRY A SAMPLE")
    
    # Handle sample file list
    samples = [
        ("sample1", "samples/sample1.wav"),
        ("sample2", "samples/sample2.wav"),
        ("sample3", "samples/sample3.wav"),
        ("sample4", "samples/sample4.wav"),
        ("sample5", "samples/sample5.wav")
    ]
    
    # Store selected path in session state
    if "selected_audio_path" not in st.session_state:
        st.session_state.selected_audio_path = None
        
    for label, path in samples:
        col_label, col_player, col_btn = st.columns([1, 4, 1])
        with col_label:
            st.write(f"**{label}**")
        with col_player:
            st.audio(path, format="audio/wav")
        with col_btn:
            if st.button("Try", key=f"btn_{label}"):
                st.session_state.selected_audio_path = path
                
    st.write("")
    run_identification = False
    
    # If the user clicks "Identify" button manually when file is uploaded
    if uploaded_file is not None:
        if st.button("Identify", key="btn_identify_upload", type="primary"):
            st.session_state.selected_audio_path = uploaded_file
            run_identification = True
            
    # Trigger automatically if a sample button is clicked
    if st.session_state.selected_audio_path is not None and uploaded_file is None:
        run_identification = True

    # Run the recognition pipeline
    if run_identification:
        active_source = st.session_state.selected_audio_path
        
        with st.spinner("Processing audio signature..."):
            t_start = time.time()
            
            # Step 1: Read audio file
            t_load_0 = time.time()
            if isinstance(active_source, str):
                y, sr = librosa.load(active_source, sr=8000)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(active_source.read())
                    temp_path = temp_file.name
                y, sr = librosa.load(temp_path, sr=8000)
                os.remove(temp_path)
            t_load = (time.time() - t_load_0) * 1000
            
            # Step 2: Compute Spectrogram
            t_spec_0 = time.time()
            peaks, S, S_db = get_peaks(y, sr)
            t_spec = (time.time() - t_spec_0) * 1000
            
            # Step 3: Hash generation
            t_hash_0 = time.time()
            hashes = generate_hashes(peaks)
            t_hash = (time.time() - t_hash_0) * 1000
            
            # Step 4: Database lookup
            t_db_0 = time.time()
            scores, best_offsets, song_offsets = match_query(hashes)
            t_db_time = (time.time() - t_db_0) * 1000
            
            # Step 5: Scoring and finding best match
            t_score_0 = time.time()
            if scores:
                best_song_idx = max(scores, key=scores.get)
                predicted_song = song_names[best_song_idx]
                match_score = scores[best_song_idx]
                best_offset = best_offsets[best_song_idx]
                
                # Calculate runner-up ratio
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 1
                ratio = match_score / max(1, runner_up_score)
            else:
                predicted_song = None
                match_score = 0
                best_offset = 0
                ratio = 1
            t_score = (time.time() - t_score_0) * 1000
            
            # Total Execution Time
            total_duration = t_spec + t_load + t_hash + t_db_time + t_score

        if predicted_song is not None:
            # 1. Performance Metrics Grid (HTML rendering matching screenshots)
            st.markdown(f"""
            <div class="metrics-container">
                <div class="metrics-header">
                    <span class="metrics-title">PERFORMANCE METRICS</span>
                    <span class="metrics-total">total {total_duration:.0f} ms</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">1. Spectrogram</div>
                        <div class="metric-value">{t_spec:.0f} ms</div>
                        <div class="metric-sub">{S_db.shape[0]}x{S_db.shape[1]}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">2. Constellation</div>
                        <div class="metric-value">{t_load:.0f} ms</div>
                        <div class="metric-sub">{len(peaks)} peaks</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">3. Hashing</div>
                        <div class="metric-value">{t_hash:.0f} ms</div>
                        <div class="metric-sub">{len(hashes):,} hashes</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">4. DB Lookup</div>
                        <div class="metric-value">{t_db_time:.0f} ms</div>
                        <div class="metric-sub">{len(song_names)} tracks</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">5. Scoring</div>
                        <div class="metric-value">{t_score:.0f} ms</div>
                        <div class="metric-sub">offset {best_offset}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. MATCH FOUND Box (HTML rendering matching screenshots)
            st.markdown(f"""
            <div class="match-container">
                <div class="match-label">MATCH FOUND</div>
                <div class="match-title">{predicted_song}</div>
                <div class="match-details">
                    cluster score <span class="match-accent">{match_score}</span> &nbsp;•&nbsp; 
                    <span class="match-accent">{ratio:.1f}x</span> the runner-up
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. CANDIDATE SCORES (Dynamic progress bar table matching screenshots)
            st.markdown('<div class="scores-container">', unsafe_allow_html=True)
            st.markdown('<div class="scores-title">CANDIDATE SCORES</div>', unsafe_allow_html=True)
            
            # Get top 5 candidate matches
            top_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            max_score = top_candidates[0][1] if top_candidates else 1
            
            for rank_idx, (s_idx, score) in enumerate(top_candidates):
                song_name = song_names[s_idx]
                pct = (score / max_score) * 100
                st.markdown(f"""
                <div class="score-row">
                    <div class="score-name">{song_name}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {pct}%;"></div>
                    </div>
                    <div class="score-value">{score}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 4. STEP 1 - FEATURE EXTRACTION
            st.markdown("""
            <div class="step-section">
                <div class="step-num">STEP 1 • FEATURE EXTRACTION</div>
                <div class="step-title">From spectrogram to constellation</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="step-desc">
            The clip was converted into a time-frequency map (left); brighter means louder at that frequency and moment. From that rich image, only the <span style="color:#2ebd9a; font-weight:bold;">{len(peaks)} most prominent peaks</span> were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.
            </div>
            """, unsafe_allow_html=True)
            
            col_spec, col_const = st.columns(2)
            
            # Plot 1A: Spectrogram (Left)
            with col_spec:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                fig.patch.set_facecolor('#121318')
                ax.set_facecolor('#121318')
                # Render spectrogram
                im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='viridis', extent=[0, len(y)/sr, 0, sr/2])
                ax.set_xlabel("time (s)", color="#8e9ca3", fontsize=9)
                ax.set_ylabel("frequency (Hz)", color="#8e9ca3", fontsize=9)
                ax.tick_params(colors="#8e9ca3", labelsize=8)
                ax.spines['bottom'].color = '#1e2029'
                ax.spines['top'].color = '#1e2029'
                ax.spines['left'].color = '#1e2029'
                ax.spines['right'].color = '#1e2029'
                st.pyplot(fig)
                
            # Plot 1B: Constellation map (Right)
            with col_const:
                fig, ax = plt.subplots(figsize=(6, 4.2))
                fig.patch.set_facecolor('#121318')
                ax.set_facecolor('#121318')
                # Render constellation peaks
                peak_times = [p[1] * 512 / sr for p in peaks]
                peak_freqs = [p[0] * sr / 1024 for p in peaks]
                ax.scatter(peak_times, peak_freqs, color='#2ebd9a', s=12, alpha=0.8)
                ax.set_xlim(0, len(y)/sr)
                ax.set_ylim(0, sr/2)
                ax.set_xlabel("time (s)", color="#8e9ca3", fontsize=9)
                ax.set_ylabel("frequency (Hz)", color="#8e9ca3", fontsize=9)
                ax.tick_params(colors="#8e9ca3", labelsize=8)
                ax.spines['bottom'].color = '#1e2029'
                ax.spines['top'].color = '#1e2029'
                ax.spines['left'].color = '#1e2029'
                ax.spines['right'].color = '#1e2029'
                st.pyplot(fig)

            # 5. STEP 2 - DATABASE SEARCH
            st.markdown("""
            <div class="step-section">
                <div class="step-num">STEP 2 • DATABASE SEARCH</div>
                <div class="step-title">Where in the song?</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="step-desc">
            The <span style="color:#2ebd9a; font-weight:bold;">{len(hashes):,} fingerprint hashes</span> were looked up against every indexed track. Below is the full fingerprint of the matched song. The highlighted window shows exactly where the query clip aligns inside the full song timeline.
            </div>
            """, unsafe_allow_html=True)
            
            # Query matched song peaks dynamically from SQLite
            conn = sqlite3.connect(DB_SQLITE)
            cursor = conn.cursor()
            cursor.execute("SELECT t_db, (hash_int >> 16) FROM fingerprints WHERE song_idx=? LIMIT 2000", (best_song_idx,))
            song_peaks = list(set(cursor.fetchall()))
            song_peaks.sort(key=lambda x: x[0])
            conn.close()
            
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor('#121318')
            ax.set_facecolor('#121318')
            
            # Plot song constellation (downsampled points in grey)
            s_step = max(1, len(song_peaks) // 1200)
            sampled_song_peaks = song_peaks[::s_step]
            s_times = [p[0] for p in sampled_song_peaks]
            s_freqs = [p[1] for p in sampled_song_peaks]
            
            ax.scatter(s_times, s_freqs, color='#3c404f', s=1.5, alpha=0.5, label='Song Fingerprint')
            
            # Highlight query window overlay
            query_length_frames = S_db.shape[1]
            ax.axvspan(best_offset, best_offset + query_length_frames, color='#2ebd9a', alpha=0.25, label='Matched Query Window')
            
            ax.set_xlabel("time (frames)", color="#8e9ca3", fontsize=9)
            ax.set_ylabel("freq bin", color="#8e9ca3", fontsize=9)
            ax.tick_params(colors="#8e9ca3", labelsize=8)
            ax.spines['bottom'].color = '#1e2029'
            ax.spines['top'].color = '#1e2029'
            ax.spines['left'].color = '#1e2029'
            ax.spines['right'].color = '#1e2029'
            ax.legend(facecolor='#121318', edgecolor='#1e2029', labelcolor='#ffffff')
            st.pyplot(fig)

            # 6. STEP 3 - THE PROOF
            st.markdown("""
            <div class="step-section">
                <div class="step-num">STEP 3 • THE PROOF</div>
                <div class="step-title">The alignment spike</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="step-desc">
            Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge: <span style="color:#e5933a; font-weight:bold;">{match_score} hashes agreed on a single offset</span>. That spike cannot be a coincidence.
            </div>
            """, unsafe_allow_html=True)
            
            # Plot 3: Alignment histogram with peak spike indicator (orange color matching screenshot)
            fig, ax = plt.subplots(figsize=(10, 4.5))
            fig.patch.set_facecolor('#121318')
            ax.set_facecolor('#121318')
            
            # Get offsets specifically for the predicted song
            predicted_offsets = song_offsets[best_song_idx]
            
            # Render histogram bins
            counts, bins, patches = ax.hist(predicted_offsets, bins=100, color='#1e2634', edgecolor='none', label='Chance Matches')
            
            # Color the consensus peak bin in solid orange
            best_bin_idx = np.digitize(best_offset, bins) - 1
            if 0 <= best_bin_idx < len(patches):
                patches[best_bin_idx].set_facecolor('#e5933a')
                patches[best_bin_idx].set_alpha(1.0)
            
            # Draw callout label for the spike
            ax.annotate(f"{match_score} hashes\nalign here", 
                        xy=(best_offset, match_score), 
                        xytext=(best_offset + 500, match_score * 0.7),
                        arrowprops=dict(facecolor='#e5933a', shrink=0.08, width=1, headwidth=6, edgecolor='none'),
                        color='#e5933a', fontweight='bold', fontsize=9)
            
            # Draw text label for chance matches
            ax.text(bins[-1] - 1500, max(counts) * 0.05, "chance\nmatches\n(noise floor)", color="#4f566b", fontsize=8, ha='center')
            
            ax.set_xlabel("time offset (database frame - query frame)", color="#8e9ca3", fontsize=9)
            ax.set_ylabel("# hashes", color="#8e9ca3", fontsize=9)
            ax.tick_params(colors="#8e9ca3", labelsize=8)
            ax.spines['bottom'].color = '#1e2029'
            ax.spines['top'].color = '#1e2029'
            ax.spines['left'].color = '#1e2029'
            ax.spines['right'].color = '#1e2029'
            st.pyplot(fig)

        else:
            st.warning("No matches found in the database. Please try another audio clip.")
            
        # Reset state to prevent infinite loops on page updates
        st.session_state.selected_audio_path = None

# ==========================================
# TAB 3: BATCH
# ==========================================
with tab_batch:
    st.write("### Identify many clips at once")
    st.markdown("""
    Upload a set of query clips. Each is identified against the **currently indexed library**, and the results are written to a standardised `results.csv` with columns `filename`, `prediction`. The `prediction` is the matched track's filename without its extension, or `none` when no candidate clears the confidence threshold.
    """)
    
    # File uploader (accept multiple files)
    batch_files = st.file_uploader("Upload batch", type=["wav", "mp3", "flac", "ogg", "m4a"], accept_multiple_files=True, label_visibility="collapsed")
    
    if batch_files:
        if st.button("Run batch", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(batch_files):
                filename = uploaded_file.name
                status_text.text(f"Processing ({idx+1}/{len(batch_files)}): {filename}...")
                
                # Write to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_path = temp_file.name
                
                try:
                    # Load audio
                    y, sr = librosa.load(temp_path, sr=8000)
                    os.remove(temp_path)
                    
                    # Extract features and match
                    peaks, _, _ = get_peaks(y, sr)
                    hashes = generate_hashes(peaks)
                    scores, _, _ = match_query(hashes)
                    
                    if scores:
                        best_song_idx = max(scores, key=scores.get)
                        # The prediction is matched track name without extension
                        prediction = song_names[best_song_idx]
                    else:
                        prediction = "none"
                        
                    results.append({
                        "filename": os.path.splitext(filename)[0],
                        "prediction": prediction
                    })
                except Exception as e:
                    results.append({
                        "filename": os.path.splitext(filename)[0],
                        "prediction": f"error: {e}"
                    })
                
                progress_bar.progress((idx + 1) / len(batch_files))
            
            status_text.text("Batch processing complete!")
            
            # Display results
            df = pd.DataFrame(results)
            st.write("#### Results Preview")
            st.dataframe(df, use_container_width=True)
            
            # Export CSV button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download results.csv",
                data=csv,
                file_name="results.csv",
                mime="text/csv"
            )
