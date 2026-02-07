import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

from subprocess import CalledProcessError, run
import whisper
from Levenshtein import ratio
import re
import json
import os

# MAIN MENU
MENU_FONT = cv2.FONT_HERSHEY_SIMPLEX
MENU_WINDOW_NAME = 'Sign Language App - Select Mode'

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ASPECT_RATIO = FRAME_WIDTH / FRAME_HEIGHT

BG_COLOR = (0, 0, 0) 
TEXT_COLOR = (240, 240, 240)
PRIMARY_COLOR = (255, 190, 0) 
SECONDARY_COLOR = (150, 150, 150) 
SUCCESS_COLOR = (0, 255, 0) 
ERROR_COLOR = (0, 0, 255) 


def wrap_text(text, font, font_scale, thickness, max_width):
    """Wraps text to fit within a specified width."""
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        
        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line.strip())
            current_line = word + " "
            
    lines.append(current_line.strip())
    return lines

# APPLICATION 1: SIGN TO TEXT/SPEECH (NEW TWO-HANDED VERSION)

def run_sign_to_text():
    print("Starting Sign-to-Text mode (Two-Handed)...")
    
    WINDOW_NAME = "Sign to Text (Two-Handed)"
    
    CONFIDENCE_THRESHOLD = 0.75 # 75%
    
    # Load the two-handed model
    try:
        with open('two_hand_model.p', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: Model file 'two_hand_model.p' not found.")
        print("Please run train_two_hand_model.py in your other project")
        print("and copy 'two_hand_model.p' into this folder.")
        print("Returning to main menu.")
        return 
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Returning to main menu.")
        return

    # Initialize MediaPipe Hands (with 2 hands)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("Returning to main menu.")
        return 

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    # Text-to-Speech Engine
    def get_tts_engine():
        """Initializes and returns a TTS engine with a slower rate."""
        try:
            engine = pyttsx3.init()
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate - 67)
            return engine
        except Exception as e:
            print(f"Error initializing TTS: {e}")
            return None

    sentence = ""
    PREDICTION_COOLDOWN_S = 1.0 
    last_word_add_time = time.time() - PREDICTION_COOLDOWN_S
    last_word_added = None 
    
    SENTENCE_BOX_HEIGHT = 120 
    SENTENCE_FONT_SCALE = 0.9 
    SENTENCE_LINE_HEIGHT = 30
    SENTENCE_PADDING = 20
    MAX_SENTENCE_LINES = (SENTENCE_BOX_HEIGHT - (SENTENCE_PADDING // 2)) // SENTENCE_LINE_HEIGHT

    print("Starting inference... Press 'Q' to quit.")

    # Main Loop (Sign-to-Text)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting mode...")
            break
        
        H_frame, W_frame, _ = frame.shape
        if H_frame != FRAME_HEIGHT or W_frame != FRAME_WIDTH:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        
        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        
        black_background = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        data_aux = np.zeros(42 * 2)
        
        predicted_sign = None
        confidence = 0.0 
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label

                x_ = []
                y_ = []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                min_x = min(x_)
                min_y = min(y_)
                
                normalized_landmarks = []
                for landmark in hand_landmarks.landmark:
                    normalized_landmarks.append(landmark.x - min_x)
                    normalized_landmarks.append(landmark.y - min_y)

                if handedness == 'Left':
                    data_aux[0:42] = normalized_landmarks
                elif handedness == 'Right':
                    data_aux[42:84] = normalized_landmarks
                
                # Draw landmarks on black background
                mp_drawing.draw_landmarks(
                    black_background, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

            # Make Prediction
            probabilities = model.predict_proba([data_aux])[0]
            confidence = np.max(probabilities)
            predicted_sign = model.classes_[np.argmax(probabilities)]
            
        show_prediction = False
        if confidence >= CONFIDENCE_THRESHOLD:
            show_prediction = True
            
        if show_prediction:
            text = f"{predicted_sign} ({confidence*100:.0f}%)"
            cv2.putText(black_background, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, PRIMARY_COLOR, 3, cv2.LINE_AA)
            
        current_time = time.time()
        
        if (current_time - last_word_add_time > PREDICTION_COOLDOWN_S):
            if show_prediction and predicted_sign != last_word_added:
                sentence += predicted_sign + " "
                last_word_add_time = current_time
                last_word_added = predicted_sign 
        
        cv2.rectangle(black_background, (0, H - SENTENCE_BOX_HEIGHT), (W, H), (255, 255, 255), -1)
        max_text_width = W - 40
        wrapped_lines = wrap_text(sentence, cv2.FONT_HERSHEY_SIMPLEX, SENTENCE_FONT_SCALE, 2, max_text_width)
        
        lines_to_draw = wrapped_lines
        if len(wrapped_lines) > MAX_SENTENCE_LINES:
            lines_to_draw = wrapped_lines[-MAX_SENTENCE_LINES:]
        
        for i, line in enumerate(lines_to_draw):
            y_pos = (H - SENTENCE_BOX_HEIGHT) + (i * SENTENCE_LINE_HEIGHT) + SENTENCE_PADDING
            cv2.putText(black_background, line, (SENTENCE_PADDING, y_pos), cv2.FONT_HERSHEY_SIMPLEX, SENTENCE_FONT_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.putText(black_background, "'Q': Quit to Menu", (W - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SECONDARY_COLOR, 2, cv2.LINE_AA)
        cv2.putText(black_background, "'C': Clear", (W - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SECONDARY_COLOR, 2, cv2.LINE_AA)
        cv2.putText(black_background, "'S': Speak", (W - 220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SECONDARY_COLOR, 2, cv2.LINE_AA)
        cv2.putText(black_background, "'Backspace': Delete", (W - 220, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SECONDARY_COLOR, 2, cv2.LINE_AA)

        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
            scale = min(win_w / W, win_h / H)
            new_w, new_h = int(W * scale), int(H * scale)
            x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
            resized_frame = cv2.resize(black_background, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[:] = BG_COLOR
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            cv2.imshow(WINDOW_NAME, canvas)
        except Exception:
            if 'black_background' in locals():
                cv2.imshow(WINDOW_NAME, black_background)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting to main menu.")
            break 
        if key == ord('c'):
            sentence = ""
            last_word_added = None
        if key == 8: 
            print("Deleting last word.")
            words = [word for word in sentence.split(' ') if word]
            if words:
                words = words[:-1]
                sentence = " ".join(words)
                if sentence: sentence += " "
            else:
                sentence = ""
            last_word_added = None
        if key == ord('s'):
            sentence_to_speak = sentence.strip()
            if sentence_to_speak:
                print(f"Speaking: {sentence_to_speak}")
                tts_engine = get_tts_engine()
                if tts_engine:
                    tts_engine.say(sentence_to_speak)
                    tts_engine.runAndWait()
                    del tts_engine

    cap.release()
    cv2.destroyWindow(WINDOW_NAME)


SAMPLE_RATE = 16000
g_whisper_model = None

def load_whisper_model():
    """Loads the Whisper model into memory."""
    global g_whisper_model
    if g_whisper_model is None:
        print("Loading Whisper 'base' model...")
        try:
            g_whisper_model = whisper.load_model('base')
            print("Whisper model loaded.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return False
    return True

def custom_load_audio(byte_data: bytes, sr=SAMPLE_RATE):
    """Converts byte data to what whisper can use."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", "-",
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"
    ]
    try:
        out = run(cmd, input=byte_data, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def process_audio(audio_bytes):
    """Loads bytes, runs Whisper, and returns text."""
    if g_whisper_model is None:
        print("Whisper model not loaded.")
        return ""
        
    try:
        audio = custom_load_audio(audio_bytes)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(g_whisper_model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(g_whisper_model, mel, options)
        return result.text
    except Exception as e:
        print(f"Error processing audio: {e}")
        return "[Audio processing failed]"

def modify_words(text, reference_data):
    """
    Modifies words so all of them are in the dictionary.
    This is now a basic typo-fixer for single words.
    """
    words = re.findall(r'\b\w+\b', text.lower().strip())
    modified_words = []
    for word in words:
        # Simple check: if the word itself is in the dictionary, keep it
        if word in reference_data:
            modified_words.append(word)
            continue
            
        # If not, try to find a close match (typo fixing)
        modified_word = None
        for reference_word in reference_data:
            similarity = ratio(word, reference_word)
            if similarity >= 0.8:
                modified_word = reference_word
                break
        
        if not modified_word is None:
            modified_words.append(modified_word)
        
    return ' '.join(modified_words)


# APPLICATION 2: TEXT/SPEECH TO SIGN (SignWave)

# Helper function to parse phrases
def parse_text_to_signs(text_input, reference_data):
    """
    Parses user text and finds the longest matching signs from the dictionary.
    Example: "hello thank you" -> ["hello", "thank you"]
    """
    words_to_play = []
    text_lower = text_input.lower()
    
    sorted_reference = sorted(reference_data, key=len, reverse=True)
    
    i = 0
    while i < len(text_lower):
        match = None
        for phrase in sorted_reference:
            if text_lower.startswith(phrase, i):
                match = phrase
                break 
        
        if match:
            words_to_play.append(match)
            i += len(match)
        else:
            i += 1
            
    print(f"Parsed text into: {words_to_play}")
    return words_to_play

def run_text_to_sign():
    print("Starting Text-to-Sign mode (SignWave)...")

    REFERENCE_JSON_PATH = 'static/json/reference.json'
    VIDEO_FOLDER = 'word videos/'  
    reference_data = [] 
    
    if os.path.exists(REFERENCE_JSON_PATH):
        with open(REFERENCE_JSON_PATH, 'r') as json_file:
            reference_data = json.load(json_file)
        print("Loaded reference.json for word modification.")
    else:
        print(f"Warning: Could not find {REFERENCE_JSON_PATH}")
        print("Word modification and video playback will not be available.")
    
    load_whisper_model()
    
    WINDOW_NAME = "Text to Sign" 
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480) 
    
    typed_sentence = ""
    display_message = "Type a sentence. Press 'Enter' to play."
    
    TEXT_BOX_X1, TEXT_BOX_Y1 = 40, 100
    TEXT_BOX_X2, TEXT_BOX_Y2 = 600, 280
    TEXT_BOX_W = TEXT_BOX_X2 - TEXT_BOX_X1 - 20 
    TEXT_PADDING = 45 
    TEXT_FONT_SCALE = 0.8 
    TEXT_LINE_HEIGHT = 25 
    TEXT_THICKNESS = 1 
    MAX_TEXT_LINES = (TEXT_BOX_Y2 - TEXT_BOX_Y1 - 10) // TEXT_LINE_HEIGHT 

    # STEP 2b: Helper function to play videos
    def play_video(video_path, window_name):
        """Opens and plays a video file in the specified window."""
        
        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"  Error: Video file not found at {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: Could not open video file {video_path}")
            return
        
        win_w, win_h = 640, 480
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        scale = min(win_w / vid_w, win_h / vid_h)
        new_w, new_h = int(vid_w * scale), int(vid_h * scale)
        x_offset = (win_w - new_w) // 2
        y_offset = (win_h - new_h) // 2
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            
            cv2.imshow(window_name, canvas)
            
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
                
        cap.release()

    while True:
        screen = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
        
        cv2.putText(screen, "Text-to-Sign Mode", (150, 60), MENU_FONT, 1, TEXT_COLOR, 2, cv2.LINE_AA)
        
        cv2.rectangle(screen, (TEXT_BOX_X1, TEXT_BOX_Y1), (TEXT_BOX_X2, TEXT_BOX_Y2), PRIMARY_COLOR, 2)
        
        wrapped_lines = wrap_text(typed_sentence, MENU_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS, TEXT_BOX_W)
        
        total_lines = len(wrapped_lines)
        lines_to_draw = wrapped_lines
        
        if total_lines > MAX_TEXT_LINES:
            lines_to_draw = wrapped_lines[-MAX_TEXT_LINES:]
            
        for i, line in enumerate(lines_to_draw):
            y = TEXT_BOX_Y1 + (i * TEXT_LINE_HEIGHT) + TEXT_LINE_HEIGHT
            cv2.putText(screen, line, (TEXT_PADDING, y), MENU_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        if wrapped_lines:
            last_line = wrapped_lines[-1]
            (last_line_width, _), _ = cv2.getTextSize(last_line, MENU_FONT, TEXT_FONT_SCALE, TEXT_THICKNESS)
            cursor_x = TEXT_PADDING + last_line_width
            
            visible_line_index = (len(lines_to_draw) - 1)
            cursor_y = TEXT_BOX_Y1 + (visible_line_index * TEXT_LINE_HEIGHT) + TEXT_LINE_HEIGHT
            
            if cursor_x > TEXT_BOX_W + TEXT_PADDING:
                cursor_x = TEXT_PADDING
                cursor_y += TEXT_LINE_HEIGHT

            if cursor_y < TEXT_BOX_Y2:
                cv2.putText(screen, cursor, (cursor_x, cursor_y), MENU_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)
        else:
            cv2.putText(screen, cursor, (TEXT_PADDING, TEXT_BOX_Y1 + TEXT_LINE_HEIGHT), MENU_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

        INSTRUCTION_Y_START_2 = TEXT_BOX_Y2 + 40 
        cv2.putText(screen, "'Enter': Play Sign Videos", (TEXT_BOX_X1, INSTRUCTION_Y_START_2), MENU_FONT, 0.7, SECONDARY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(screen, "'Esc': Clear Text", (TEXT_BOX_X1, INSTRUCTION_Y_START_2 + 30), MENU_FONT, 0.7, SECONDARY_COLOR, 1, cv2.LINE_AA)
        cv2.putText(screen, "'Q': Quit to Menu", (TEXT_BOX_X1, INSTRUCTION_Y_START_2 + 60), MENU_FONT, 0.7, SECONDARY_COLOR, 1, cv2.LINE_AA)
        
        cv2.putText(screen, display_message, (TEXT_BOX_X1, INSTRUCTION_Y_START_2 + 110), MENU_FONT, 0.8, PRIMARY_COLOR, 1, cv2.LINE_AA)

        # Letterboxing logic for Mode 2
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
            scale = min(win_w / FRAME_WIDTH, win_h / FRAME_HEIGHT)
            new_w, new_h = int(FRAME_WIDTH * scale), int(FRAME_HEIGHT * scale)
            x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
            resized_frame = cv2.resize(screen, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[:] = BG_COLOR
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            cv2.imshow(WINDOW_NAME, canvas)
        except Exception:
            if 'screen' in locals():
                cv2.imshow(WINDOW_NAME, screen)
        
        key = cv2.waitKey(100) & 0xFF 
        
        if key == ord('q'): 
            print("Quitting to main menu.")
            break
        
        elif key == 27: 
            print("Clearing text.")
            typed_sentence = ""
            display_message = "Text cleared."
        
        elif key == 8: 
            typed_sentence = typed_sentence[:-1]
            
        elif key == 13: 
            if not typed_sentence:
                display_message = "Please type a sentence first."
                continue
                
            print(f"Processing sentence: {typed_sentence}")
            
            modified_text = typed_sentence
            if reference_data:
                words_in_sentence = typed_sentence.split(' ')
                fixed_words = []
                for word in words_in_sentence:
                    fixed_words.append(modify_words(word, reference_data))
                modified_text = " ".join(fixed_words)
                print(f"Modified text (typo fix): {modified_text}")

            # 2. Parse the text into signs (single and multi-word)
            words_to_play = parse_text_to_signs(modified_text, reference_data)
            
            # STEP 2c: This is the implemented video logic
            display_message = f"Playing: {' '.join(words_to_play)}"
            
            screen = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
            cv2.putText(screen, display_message, (40, 350), MENU_FONT, 0.8, PRIMARY_COLOR, 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, screen)
            cv2.waitKey(1) 
            
            for word in words_to_play:
                
                video_filename = f"{word}.mp4"
                video_path = os.path.join(VIDEO_FOLDER, video_filename)
                    
                if os.path.exists(video_path): 
                    print(f"  Playing video for: '{word}' (Path: {video_path})")
                    play_video(video_path, WINDOW_NAME)
                else:
                    print(f"  No video found for: '{word}' (Path: {video_path})")
                    screen = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
                    cv2.putText(screen, f"Video not found for: '{word}'", (100, 240), MENU_FONT, 1, ERROR_COLOR, 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW_NAME, screen)
                    cv2.waitKey(1000) 
            
            print("Playback finished.")
            display_message = "Playback finished. Type a new sentence."
        
        elif 32 <= key <= 126: 
            typed_sentence += chr(key)
            
    cv2.destroyWindow(WINDOW_NAME)


# MAIN MENU

if __name__ == "__main__":
    
    cv2.namedWindow(MENU_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MENU_WINDOW_NAME, 640, 480)
    
    selected_option = 1
    
    while True:
        
        menu_screen = np.full((480, 640, 3), BG_COLOR, dtype=np.uint8)
        
        title = "Select Mode:"
        (w, h), _ = cv2.getTextSize(title, MENU_FONT, 1.2, 2)
        cv2.putText(menu_screen, title, ((FRAME_WIDTH - w) // 2, 100), MENU_FONT, 1.2, TEXT_COLOR, 2, cv2.LINE_AA)
        
        # Option 1
        color1 = PRIMARY_COLOR if selected_option == 1 else TEXT_COLOR
        text1 = "1: Sign to Text/Speech"
        if selected_option == 1:
            text1 = "> 1: Sign to Text/Speech <"
        (w, h), _ = cv2.getTextSize(text1, MENU_FONT, 1, 2)
        cv2.putText(menu_screen, text1, ((FRAME_WIDTH - w) // 2, 200), MENU_FONT, 1, color1, 2, cv2.LINE_AA)
        
        # Option 2
        color2 = PRIMARY_COLOR if selected_option == 2 else TEXT_COLOR
        text2 = "2: Text/Speech to Sign"
        if selected_option == 2:
            text2 = "> 2: Text/Speech to Sign <"
        (w, h), _ = cv2.getTextSize(text2, MENU_FONT, 1, 2)
        cv2.putText(menu_screen, text2, ((FRAME_WIDTH - w) // 2, 270), MENU_FONT, 1, color2, 2, cv2.LINE_AA)

        # Option 3 (Quit)
        colorQ = PRIMARY_COLOR if selected_option == 3 else TEXT_COLOR
        textQ = "Q: Quit Application"
        if selected_option == 3:
            textQ = "> Q: Quit Application <"
        (w, h), _ = cv2.getTextSize(textQ, MENU_FONT, 1, 2)
        cv2.putText(menu_screen, textQ, ((FRAME_WIDTH - w) // 2, 340), MENU_FONT, 1, colorQ, 2, cv2.LINE_AA)
        
        # Letterboxing logic for Main Menu
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(MENU_WINDOW_NAME)
            if win_w <= 0 or win_h <= 0: raise Exception("Window minimized")
            scale = min(win_w / FRAME_WIDTH, win_h / FRAME_HEIGHT)
            new_w, new_h = int(FRAME_WIDTH * scale), int(FRAME_HEIGHT * scale)
            x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
            resized_frame = cv2.resize(menu_screen, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[:] = BG_COLOR
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
            cv2.imshow(MENU_WINDOW_NAME, canvas)
        except Exception as e: 
            if 'menu_screen' in locals():
                cv2.imshow(MENU_WINDOW_NAME, menu_screen)
        
        key = cv2.waitKey(50) 
        
        if key == 2490368: 
            selected_option -= 1
            if selected_option < 1:
                selected_option = 3
        elif key == 2621440:
            selected_option += 1
            if selected_option > 3:
                selected_option = 1
        
        # Standard keys (for shortcuts)
        key_char = key & 0xFF
        
        if key_char == ord('1'):
            selected_option = 1
        elif key_char == ord('2'):
            selected_option = 2
        elif key_char == ord('q'):
            selected_option = 3
            
        if key_char == 13:
            if selected_option == 1:
                print("Selected: 1")
                run_sign_to_text() 
            elif selected_option == 2:
                print("Selected: 2")
                run_text_to_sign() 
            elif selected_option == 3:
                print("Selected: Q. Exiting.")
                break
        
        if key_char == ord('q'):
             print("Selected: Q. Exiting.")
             break
        
        try:
            if cv2.getWindowProperty(MENU_WINDOW_NAME, cv2.WND_PROP_AUTOSIZE) == -1:
                print("Window closed. Exiting.")
                break
        except:
            continue
            
    cv2.destroyAllWindows()