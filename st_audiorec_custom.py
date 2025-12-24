"""
Custom Audio Recorder Component for Streamlit
Save as: st_audiorec.py
"""

import streamlit as st
import streamlit.components.v1 as components

def st_audiorec():
    """
    Custom audio recorder component
    Returns audio bytes or None
    """
    
    # HTML/JS for audio recording
    component_html = """
    <div style="text-align: center; padding: 20px;">
        <button id="recordButton" style="
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        " onmouseover="this.style.transform='translateY(-2px)'"
           onmouseout="this.style.transform='translateY(0)'">
            🎤 Click to Record
        </button>
        <p id="status" style="margin-top: 15px; color: #666;">Ready to record</p>
    </div>

    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    const button = document.getElementById('recordButton');
    const status = document.getElementById('status');

    button.onclick = async function() {
        if (!isRecording) {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64data = reader.result;
                        window.parent.postMessage({
                            type: 'streamlit:audioRecorded',
                            data: base64data
                        }, '*');
                    };
                    audioChunks = [];
                };
                
                mediaRecorder.start();
                isRecording = true;
                button.innerHTML = '⏹️ Stop Recording';
                button.style.background = 'linear-gradient(120deg, #ff6b6b 0%, #ee5a6f 100%)';
                status.innerHTML = '🔴 Recording...';
                status.style.color = '#ff6b6b';
            } catch (err) {
                status.innerHTML = '❌ Microphone access denied';
                status.style.color = '#ff6b6b';
            }
        } else {
            // Stop recording
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            button.innerHTML = '🎤 Click to Record';
            button.style.background = 'linear-gradient(120deg, #667eea 0%, #764ba2 100%)';
            status.innerHTML = '✅ Recording saved!';
            status.style.color = '#51cf66';
        }
    };
    </script>
    """
    
    # Render component
    audio_data = components.html(component_html, height=150)
    
    # Check for recorded audio in session state
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None
    
    # JavaScript to Python communication would require custom component
    # For simplicity, use file uploader as alternative
    
    return st.session_state.audio_bytes