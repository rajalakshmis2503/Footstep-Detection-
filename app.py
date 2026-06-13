import streamlit as st
import os
import tempfile
import time
import numpy as np
from AFPILD_Predict import predict_locations, plot_footstep_locations
from Anomaly_Detect import detect_anomaly
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import wave
import struct

# Set animation embed limit to 50MB
plt.rcParams['animation.embed_limit'] = 50

# Initialize session state
if 'stage' not in st.session_state:
    st.session_state.stage = 1
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'tmp_file_path' not in st.session_state:
    st.session_state.tmp_file_path = None
if 'anomaly_detected' not in st.session_state:
    st.session_state.anomaly_detected = None
if 'anomaly_details' not in st.session_state:
    st.session_state.anomaly_details = None

def analyze_movement_patterns(predictions):
    """Analyze movement patterns for anomalies"""
    if not predictions or len(predictions) < 2:
        return None
    
    # Calculate movement metrics
    speeds = []
    step_distances = []
    for i in range(len(predictions)-1):
        dx = predictions[i+1]['x'] - predictions[i]['x']
        dy = predictions[i+1]['y'] - predictions[i]['y']
        dt = predictions[i+1]['time'] - predictions[i]['time']
        distance = np.sqrt(dx*dx + dy*dy)
        speed = distance/dt if dt > 0 else 0
        speeds.append(speed)
        step_distances.append(distance)
    
    avg_speed = np.mean(speeds)
    avg_step = np.mean(step_distances)
    std_speed = np.std(speeds)
    std_step = np.std(step_distances)
    
    # Detect anomalies in movement patterns
    anomalies = {
        'irregular_speed': [],
        'unusual_steps': [],
        'sudden_turns': []
    }
    
    # Check for irregular speeds (outside 2 standard deviations)
    for i, speed in enumerate(speeds):
        if abs(speed - avg_speed) > 2 * std_speed:
            anomalies['irregular_speed'].append(i+1)
    
    # Check for unusual step distances
    for i, dist in enumerate(step_distances):
        if abs(dist - avg_step) > 2 * std_step:
            anomalies['unusual_steps'].append(i+1)
    
    # Check for sudden direction changes
    for i in range(len(predictions)-2):
        v1x = predictions[i+1]['x'] - predictions[i]['x']
        v1y = predictions[i+1]['y'] - predictions[i]['y']
        v2x = predictions[i+2]['x'] - predictions[i+1]['x']
        v2y = predictions[i+2]['y'] - predictions[i+1]['y']
        
        # Calculate angle between vectors
        dot_product = v1x*v2x + v1y*v2y
        mag1 = np.sqrt(v1x*v1x + v1y*v1y)
        mag2 = np.sqrt(v2x*v2x + v2y*v2y)
        if mag1 > 0 and mag2 > 0:
            angle = np.arccos(dot_product/(mag1*mag2))
            if abs(angle) > np.pi/2:  # More than 90 degrees turn
                anomalies['sudden_turns'].append(i+1)
    
    return anomalies

def reset_app():
    st.session_state.stage = 1
    st.session_state.predictions = None
    st.session_state.anomaly_detected = None
    st.session_state.anomaly_details = None
    if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
        os.unlink(st.session_state.tmp_file_path)
    st.session_state.tmp_file_path = None

def validate_wav_file(file_path):
    """Validate and get information about a WAV file"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            return {
                'channels': channels,
                'sample_width': sample_width,
                'frame_rate': frame_rate,
                'frames': frames,
                'duration': frames / float(frame_rate),
                'valid': True
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def read_wav_file(file_path):
    """Read WAV file using wave module and convert to numpy array"""
    with wave.open(file_path, 'rb') as wav_file:
        # Get file parameters
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read raw data
        raw_data = wav_file.readframes(n_frames)
        
        # Convert raw data to numpy array
        if sample_width == 2:
            data = np.frombuffer(raw_data, dtype=np.int16)
        elif sample_width == 4:
            data = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Normalize to float32 between -1 and 1
        data = data.astype(np.float32) / (2**(8 * sample_width - 1))
        
        # Reshape to [frames, channels]
        data = data.reshape(-1, channels)
        
        return data, frame_rate

# Page 1: File Upload
def show_upload_page():
    st.title("Audio Footstep Localization")
    st.markdown("""
        ### Welcome to the Audio Footstep Localization System
        This tool helps you analyze audio recordings to:
        - Detect individual footsteps
        - Predict the location of each footstep
        - Visualize movement patterns
        
    """)
    
    # Add custom CSS
    st.markdown("""
        <style>
        .file-details {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file-detail-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #dee2e6;
        }
        .file-detail-item:last-child {
            border-bottom: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav'], 
        help="Upload a WAV file")
    
    if uploaded_file is not None:
        try:
            # First save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_upload:
                file_content = uploaded_file.getvalue()
                # Check if file content starts with RIFF header
                if not file_content.startswith(b'RIFF'):
                    raise ValueError("Invalid WAV file: Missing RIFF header")
                tmp_upload.write(file_content)
                tmp_upload_path = tmp_upload.name

            # Validate the WAV file
            wav_info = validate_wav_file(tmp_upload_path)
            if not wav_info['valid']:
                raise ValueError(f"Invalid WAV file: {wav_info['error']}")

            st.info(f"""
            File details:
            - Channels: {wav_info['channels']}
            - Sample width: {wav_info['sample_width']} bytes
            - Frame rate: {wav_info['frame_rate']} Hz
            - Duration: {wav_info['duration']:.2f} seconds
            """)

            # Read the audio data using wave instead of soundfile
            audio_data, sr = read_wav_file(tmp_upload_path)
            
            # Audio data is already in the correct shape [frames, channels]
            # Just transpose to get [channels, frames] as expected by the model
            audio_data = audio_data.T
            
            # Save processed audio to a new temporary file using wave directly
            with wave.open(tmp_processed_path := tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name, 'wb') as wav_out:
                wav_out.setnchannels(4)
                wav_out.setsampwidth(2)  # 16-bit
                wav_out.setframerate(sr)
                # Convert back to int16
                audio_int16 = (audio_data.T * 32767).astype(np.int16)
                wav_out.writeframes(audio_int16.tobytes())
            
            st.session_state.tmp_file_path = tmp_processed_path
            
            # Validate the processed file
            processed_wav_info = validate_wav_file(st.session_state.tmp_file_path)
            if not processed_wav_info['valid']:
                raise ValueError(f"Error in processed WAV file: {processed_wav_info['error']}")
            
            # Clean up the initial temporary file
            os.unlink(tmp_upload_path)
            
            st.success("✅ File uploaded and processed successfully!")
            
            # Center the process button
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                if st.button("Process Audio", use_container_width=True):
                    st.session_state.stage = 2
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing audio file: {str(e)}")
            if st.session_state.tmp_file_path and os.path.exists(st.session_state.tmp_file_path):
                os.unlink(st.session_state.tmp_file_path)
            if 'tmp_upload_path' in locals() and os.path.exists(tmp_upload_path):
                os.unlink(tmp_upload_path)
            if 'tmp_processed_path' in locals() and os.path.exists(tmp_processed_path):
                os.unlink(tmp_processed_path)

# Page 2: Processing
def show_processing_page():
    st.title("Processing Audio")
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate processing steps
    steps = [
        "Loading audio file...",
        "Checking for anomalies...",
        "Extracting features...",
        "Making predictions...",
        "Analyzing movement patterns...",
        "Generating visualization..."
    ]
    
    status_text.text(steps[0])
    progress_bar.progress(1 / len(steps))
    time.sleep(1)
    
    try:
        # Check for anomalies
        status_text.text(steps[1])
        progress_bar.progress(2 / len(steps))
        st.session_state.anomaly_detected = detect_anomaly(st.session_state.tmp_file_path)

        # Make predictions
        status_text.text(steps[2])
        progress_bar.progress(3 / len(steps))
        time.sleep(1)

        status_text.text(steps[3])
        progress_bar.progress(4 / len(steps))
        predictions = predict_locations("super_duper_model.h5", st.session_state.tmp_file_path)
        st.session_state.predictions = predictions
        
        # Analyze movement patterns
        status_text.text(steps[4])
        progress_bar.progress(5 / len(steps))
        st.session_state.anomaly_details = analyze_movement_patterns(predictions)

        status_text.text(steps[5])
        progress_bar.progress(6 / len(steps))
        time.sleep(1)
        # Move to results page
        st.session_state.stage = 3
        st.rerun()
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        if st.button("Try Again"):
            reset_app()
            st.rerun()

# Page 3: Results
def show_results_page():
    st.title("Footstep Localization Results")
    
    if st.session_state.predictions:
        # Create an expandable section for anomaly alerts
        with st.expander("🚨 Anomaly Detection Results", expanded=True):
            if st.session_state.anomaly_detected:
                st.error("  Anomaly Detected: Unusual footstep pattern detected in the audio", icon="⚠️")
                
                # Display detailed anomaly analysis
                if st.session_state.anomaly_details:
                    anomalies = st.session_state.anomaly_details
                    
                    # Display irregular speeds
                    if anomalies['irregular_speed']:
                        st.warning(f"🏃 Irregular Movement Speed detected at steps: {', '.join(map(str, anomalies['irregular_speed']))}")
                    
                    # Display unusual steps
                    if anomalies['unusual_steps']:
                        st.warning(f"👣 Unusual Step Distances detected at steps: {', '.join(map(str, anomalies['unusual_steps']))}")
                    
                    # Display sudden turns
                    if anomalies['sudden_turns']:
                        st.warning(f"↩️ Sudden Direction Changes detected at steps: {', '.join(map(str, anomalies['sudden_turns']))}")
                    
                    # Add recommendations
                    st.info("💡 Recommendations:")
                    st.markdown("""
                        - Review the highlighted steps in the visualization
                        - Check for potential security concerns
                        - Consider reviewing surveillance footage for these timestamps
                        - Document these anomalies for pattern analysis
                    """)
            else:
                st.success("✅ No Anomalies Detected: Normal footstep pattern confirmed")

        # Add metrics at the top
        metrics_cols = st.columns(5)  # Changed from 4 to 5 to include anomaly status
        total_steps = len(st.session_state.predictions)
        total_distance = sum(
            ((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)**0.5 
            for p1, p2 in zip(st.session_state.predictions[:-1], st.session_state.predictions[1:])
        )
        duration = st.session_state.predictions[-1]['time'] - st.session_state.predictions[0]['time']
        avg_speed = total_distance / duration if duration > 0 else 0

        # Calculate most common subject
        subjects = [p['subject'] for p in st.session_state.predictions]
        most_common_subject = max(set(subjects), key=subjects.count)
        subject_confidence = (subjects.count(most_common_subject) / len(subjects)) * 100

        # Create a styled info box for subject identification
        subject_info_html = f"""
            <div style='
                padding: 20px;
                background-color: #e6f3ff;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            '>
                <h2 style='margin: 0;'>
                    Subject Identified: Person {most_common_subject}
                </h2>
                <p style='margin: 5px 0 0 0;'>
                    Confidence: {subject_confidence:.1f}%
                </p>
            </div>
        """
        st.markdown(subject_info_html, unsafe_allow_html=True)

        with metrics_cols[0]:
            st.metric("Total Steps", f"{total_steps}")
        with metrics_cols[1]:
            st.metric("Total Distance", f"{total_distance:.2f} m")
        with metrics_cols[2]:
            st.metric("Duration", f"{duration:.2f} s")
        with metrics_cols[3]:
            st.metric("Average Speed", f"{avg_speed:.2f} m/s")
        with metrics_cols[4]:
            st.metric("Anomaly Detected", st.session_state.anomaly_detected)

        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Visualization", "📋 Detailed Data"])
        
        with tab1:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Add visualization controls
            viz_cols = st.columns([3, 1])
            with viz_cols[1]:
                st.markdown("### Visualization Controls")
                show_arrows = st.checkbox("Show Movement Arrows", value=True)
                show_colorbar = st.checkbox("Show Time Colorbar", value=True)
                marker_size = st.slider("Marker Size", 50, 200, 100)
                
            with viz_cols[0]:
                # Plot the results with custom settings
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                
                # Extract coordinates
                x_coords = [p['x'] for p in st.session_state.predictions]
                y_coords = [p['y'] for p in st.session_state.predictions]
                times = [p['time'] for p in st.session_state.predictions]
                
                # Create scatter plot
                scatter = ax.scatter(x_coords, y_coords, 
                                   c=times if show_colorbar else None,
                                   cmap='viridis',
                                   s=marker_size, 
                                   alpha=0.6)
                
                if show_colorbar:
                    plt.colorbar(scatter, label='Time (seconds)')
                
                if show_arrows:
                    for i in range(len(x_coords)-1):
                        ax.arrow(x_coords[i], y_coords[i],
                                x_coords[i+1]-x_coords[i], y_coords[i+1]-y_coords[i],
                                color='gray', alpha=0.3, 
                                head_width=0.05, length_includes_head=True)
                
                ax.set_xlabel('X Coordinate (m)')
                ax.set_ylabel('Y Coordinate (m)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tab2:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            # Add search and filter options
            search_col, filter_col = st.columns([2, 1])
            with search_col:
                search = st.text_input("🔍 Search in data")
            with filter_col:
                sort_by = st.selectbox("Sort by", ["Time (s)", "X (m)", "Y (m)"])
            
            # Prepare and filter data
            data = [{
                'Time (s)': f"{p['time']:.2f}",
                'X (m)': f"{p['x']:.2f}",
                'Y (m)': f"{p['y']:.2f}",
                'Step #': i+1
            } for i, p in enumerate(st.session_state.predictions)]
            
            # Apply search filter
            if search:
                data = [d for d in data if any(search.lower() in str(v).lower() for v in d.values())]
            
            # Apply sorting
            if sort_by:
                data = sorted(data, key=lambda x: float(x[sort_by]) if sort_by != "Subject" else x[sort_by])
            
            # Display the table with styling
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Step #": st.column_config.NumberColumn(
                        "Step #",
                        help="Step sequence number",
                        format="%d"
                    ),
                    "Time (s)": st.column_config.NumberColumn(
                        "Time (s)",
                        help="Time of footstep detection",
                        format="%.2f"
                    ),
                    "X (m)": st.column_config.NumberColumn(
                        "X (m)",
                        help="X coordinate in meters",
                        format="%.2f"
                    ),
                    "Y (m)": st.column_config.NumberColumn(
                        "Y (m)",
                        help="Y coordinate in meters",
                        format="%.2f"
                    )
                }
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

    # Add export options
    if st.session_state.predictions:
        st.markdown("### Export Options")
        export_cols = st.columns(3)
        with export_cols[0]:
            if st.button("Export to CSV"):
                # Add CSV export functionality here
                st.download_button(
                    label="Download CSV",
                    data="\n".join([
                        "Time (s),X (m),Y (m)",
                        *[f"{p['time']:.2f},{p['x']:.2f},{p['y']:.2f}"
                          for p in st.session_state.predictions]
                    ]),
                    file_name="footstep_predictions.csv",
                    mime="text/csv"
                )
    
    # Add button to process another file
    st.sidebar.button("Process Another File", on_click=reset_app)

# Main app logic
def main():
    st.set_page_config(page_title="Audio Footstep Localization", layout="wide")
    
    if st.session_state.stage == 1:
        show_upload_page()
    elif st.session_state.stage == 2:
        show_processing_page()
    elif st.session_state.stage == 3:
        show_results_page()

if __name__ == "__main__":
    main() 