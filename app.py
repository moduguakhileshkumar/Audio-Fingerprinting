import streamlit as st
import os
import pickle
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
    static_ffmpeg.add_paths()

init_ffmpeg()

# Load fingerprint database
@st.cache_resource
def load_database():
    db_path = "fingerprint_db.pkl"
    if not os.path.exists(db_path):
        return None
    with open(db_path, "rb") as f:
        return pickle.load(f)

with st.spinner("⏳ Loading fingerprint database (310MB)... This may take 20-30 seconds on the first run. Please wait."):
    db_data = load_database()

if db_data is None:
    st.error("Fingerprint database not found! Please make sure 'fingerprint_db.pkl' is built and in the app directory.")
    st.stop()

database = db_data["database"]
song_names = db_data["song_names"]

# Pre-compile song information for Tab 1 (LIBRARY)
@st.cache_resource
def compile_library_info(_database, song_names):
    song_hashes_count = defaultdict(int)
    song_peaks = defaultdict(list)
    
    for h_key, occurrences in _database.items():
        f1, f2, dt = h_key
        for song_idx, t_db in occurrences:
            song_hashes_count[song_idx] += 1
            # Keep coordinates for scatter plot
            song_peaks[song_idx].append((t_db, f1))
            
    compiled_info = {}
    for song_idx, name in enumerate(song_names):
        # Sort and unique peaks
        peaks = list(set(song_peaks[song_idx]))
        peaks.sort(key=lambda x: x[0])
        compiled_info[song_idx] = {
            "name": name,
            "hashes_count": song_hashes_count[song_idx],
            "peaks": peaks
        }
    return compiled_info

library_info = compile_library_info(database, song_names)

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

def match_query(q_hashes):
    match_counts = defaultdict(int)
    song_offsets = defaultdict(list)
    
    for h_key, t_q in q_hashes:
        if h_key in database:
            for song_idx, t_db in database[h_key]:
                offset = t_db - t_q
                match_counts[(song_idx, offset)] += 1
                song_offsets[song_idx].append(offset)
                
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
    
    # 8 songs displayed in the professor's screenshots
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
    
    # Map visible song names to their indices in database
    visible_song_indices = []
    for s_name in target_visible_songs:
        idx = next((i for i, name in enumerate(song_names) if name.lower() == s_name.lower()), None)
        if idx is not None:
            visible_song_indices.append(idx)
            
    # Add a toggle to show/hide other songs
    cols = st.columns(4)
    
    # Draw cards for the 8 visible songs
    for i, idx in enumerate(visible_song_indices):
        col = cols[i % 4]
        song_info = library_info[idx]
        
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
            
            # Downsample peaks to 400 points to keep plotting extremely fast
            peaks = song_info['peaks']
            step = max(1, len(peaks) // 400)
            sampled_peaks = peaks[::step]
            
            times = [p[0] for p in sampled_peaks]
            freqs = [p[1] for p in sampled_peaks]
            
            ax.scatter(times, freqs, color=card_colors[i % len(card_colors)], s=1.5, alpha=0.6)
            ax.set_xlim(0, max(times) if times else 6000)
            ax.set_ylim(0, 512)
            ax.axis('off')
            
            st.pyplot(fig)
            st.write("") # Spacer

    # Toggle for other songs in the database
    with st.expander("Show other songs in database"):
        other_songs = []
        for idx, name in enumerate(song_names):
            if idx not in visible_song_indices:
                other_songs.append({
                    "Song Index": idx + 1,
                    "Song Title": name,
                    "Total Hashes": f"{library_info[idx]['hashes_count']:,}"
                })
        st.dataframe(pd.DataFrame(other_songs), use_container_width=True)

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
            
            # Plot 2: Large song constellation map with highlight overlay
            song_info = library_info[best_song_idx]
            song_peaks = song_info["peaks"]
            
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
