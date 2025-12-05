import sys
import os
import threading
import time
import datetime
import webbrowser
import tkinter as tk
from typing import Optional

import customtkinter as ctk
import cv2
import mediapipe as mp
import pyperclip
import google.generativeai as genai
from pynput import mouse, keyboard
from PIL import Image, ImageTk

# --- Configuration ---
GENAI_API_KEY = ""  # User must replace this
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- Constants ---
DRAG_THRESHOLD = 50  # pixels
GESTURE_CONFIDENCE = 0.7
OUTPUT_DIR = "saved_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- AI Client ---
class GeminiClient:
    def __init__(self, api_key):
        if api_key and api_key != "YOUR_API_KEY_HERE":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.active = True
        else:
            self.active = False
            print("Warning: No valid API Key provided.")

    def explain_text(self, text):
        if not self.active:
            return "Error: API Key not set. Please configure GENAI_API_KEY in main.py."
        
        prompt = (
            "ユーザーは大学生です。提供されたテキストを必ず日本語でわかりやすく解説・要約してください。"
            "入力が英語であっても日本語で出力してください。\n\n"
            f"テキスト: {text}"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating content: {e}"

    def quick_explain(self, text):
        if not self.active:
            return "Error: API Key not set."
        
        prompt = (
            "以下の用語を1〜2文の日本語で端的に説明してください。\n"
            f"用語: {text}"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"

    def generate_html_report(self, text):
        if not self.active:
            return "Error: API Key not set."
        
        prompt = (
            "以下のテキストに関する詳細なHTMLレポートを作成してください。\n"
            "要件:\n"
            "1. タイトルと概要を含めること。\n"
            "2. 重要なポイントを比較表やリストで整理すること。\n"
            "3. 関連する信頼できる外部サイトへのリンクを3つ以上含めること（<a>タグを使用）。\n"
            "4. デザインはシンプルで見やすくすること（CSSを埋め込む）。\n"
            "5. 全体的に日本語で記述すること。\n\n"
            f"テキスト: {text}"
        )
        try:
            response = self.model.generate_content(prompt)
            # Cleanup markdown code blocks if present
            html = response.text.replace("```html", "").replace("```", "")
            return html
        except Exception as e:
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"

# --- Gesture Recognition ---
class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=GESTURE_CONFIDENCE,
            min_tracking_confidence=GESTURE_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        gesture = None
        landmarks = None

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            gesture = self._classify_gesture(landmarks)
        
        return gesture, landmarks

    def _classify_gesture(self, landmarks):
        # Simple heuristic based gesture recognition
        # Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
        tips = [4, 8, 12, 16, 20]
        pip = [2, 6, 10, 14, 18] # PIP joints for checking if finger is open
        
        # Get coordinates
        lms = landmarks.landmark
        
        fingers = []
        # Thumb (check x for handedness, but assuming right hand or general open/close for simplicity)
        # For simplicity, comparing tip to IP joint (3) might be better, or x diff
        # Using a simpler approach: check if tip is above PIP (y is inverted in screen coords)
        
        # Index
        fingers.append(1 if lms[8].y < lms[6].y else 0)
        # Middle
        fingers.append(1 if lms[12].y < lms[10].y else 0)
        # Ring
        fingers.append(1 if lms[16].y < lms[14].y else 0)
        # Pinky
        fingers.append(1 if lms[20].y < lms[18].y else 0)
        
        # Thumb is tricky without handedness. Let's check if it's extended away from palm
        # Simple check: thumb tip x is far from pinky base x? 
        # Or just check if thumb tip is "above" (lower y) than thumb IP?
        # Let's use a standard count.
        
        # Pinch detection (Thumb + Index close)
        distance_thumb_index = ((lms[4].x - lms[8].x)**2 + (lms[4].y - lms[8].y)**2)**0.5
        if distance_thumb_index < 0.05:
            return "PINCH"

        total_fingers = sum(fingers)
        
        # Thumb check (approximate)
        thumb_open = 0
        if lms[4].x < lms[3].x: # Right hand palm facing camera? 
             # This is hard to generalize without handedness. 
             # Let's rely on the 4 fingers for main gestures and assume thumb is open for 5
             pass
        
        # Logic for specific gestures
        if total_fingers == 0 and distance_thumb_index > 0.05:
            return "FIST" # Not used
        
        if fingers == [1, 1, 0, 0]: # Index and Middle
            return "PEACE"
        
        if fingers == [0, 1, 1, 1]: # OK Sign (Thumb and Index touching is Pinch, but if others are open...)
             # OK is usually Thumb+Index touching, others open.
             if distance_thumb_index < 0.05:
                 return "OK"
        
        if fingers == [0, 0, 0, 0]: # All closed
             # Check thumb for Thumbs Up
             # Thumb tip y < Thumb IP y and others closed
             if lms[4].y < lms[3].y and lms[4].y < lms[8].y:
                 return "GOOD"

        return None

# --- UI Components ---

class QuickInfoWindow(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.9)
        self.configure(fg_color="#333333")
        
        self.label = ctk.CTkLabel(
            self, 
            text="Loading...", 
            font=("Meiryo", 12), 
            text_color="white",
            wraplength=300,
            justify="left"
        )
        self.label.pack(padx=10, pady=10)
        self.withdraw()

    def show_text(self, text, x, y):
        self.label.configure(text=text)
        # Adjust geometry to not go off screen? For now simple offset.
        self.geometry(f"+{x+20}+{y+20}")
        self.deiconify()

    def hide(self):
        self.withdraw()

class OverlayButton(ctk.CTkToplevel):
    def __init__(self, on_settings_click, on_search_hover_start, on_search_hover_end):
        super().__init__()
        self.on_settings_click = on_settings_click
        self.on_search_hover_start = on_search_hover_start
        self.on_search_hover_end = on_search_hover_end
        
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.geometry("150x50")
        # Windows workaround for transparency
        self.configure(fg_color="white")
        self.attributes("-transparentcolor", "white")
        
        self.frame = ctk.CTkFrame(self, fg_color="transparent")
        self.frame.pack(fill="both", expand=True)

        # Search Button (Hover for Quick Info)
        self.btn_search = ctk.CTkButton(
            self.frame, 
            text="🔍", 
            width=45, 
            height=20,
            corner_radius=0,
            fg_color="#1f538d",
            hover_color="#14375e",
            command=lambda: None # No click action, hover only
        )
        self.btn_search.pack(side="left", padx=2)
        
        self.btn_search.bind("<Enter>", self.on_enter_search)
        self.btn_search.bind("<Leave>", self.on_leave_search)

        # Settings Button (Click for Full Mode)
        self.btn_settings = ctk.CTkButton(
            self.frame, 
            text="AI説明", 
            font=("Meiryo", 12, "bold"),
            width=110, 
            height=20,
            corner_radius=0,
            fg_color="#555555",
            hover_color="#333333",
            command=self.on_click_settings
        )
        self.btn_settings.pack(side="right", padx=2)

        self.withdraw() # Hidden initially
        self.x = 0
        self.y = 0

    def show_at(self, x, y):
        self.geometry(f"+{x}+{y}")
        self.x = x
        self.y = y
        self.deiconify()

    def is_inside(self, check_x, check_y):
        if self.state() != "normal":
            return False
        # Check if click is within the 150x50 box at (self.x, self.y)
        return (self.x <= check_x <= self.x + 150) and (self.y <= check_y <= self.y + 50)

    def on_click_settings(self):
        self.withdraw()
        self.on_settings_click()

    def on_enter_search(self, event):
        self.on_search_hover_start()

    def on_leave_search(self, event):
        self.on_search_hover_end()

class ResultWindow(ctk.CTkToplevel):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.title("AI Explanation")
        self.geometry("800x600")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar (Camera & Controls)
        self.sidebar = ctk.CTkFrame(self, width=750, corner_radius=0)
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)

        self.sidebar.grid_rowconfigure(4, weight=1)

        # Camera Label Removed
        # self.cam_label = ctk.CTkLabel(self.sidebar, text="Camera OFF", width=240, height=180, fg_color="black")
        # self.cam_label.grid(row=0, column=0, padx=5, pady=5)

        self.cam_btn = ctk.CTkButton(self.sidebar, text="📷 ON", command=self.toggle_camera, fg_color="green")
        self.cam_btn.grid(row=0, column=1, padx=10, pady=5)

        self.btn_save = ctk.CTkButton(self.sidebar, text="保存して閉じる (👍)", command=self.save_and_close)
        self.btn_save.grid(row=0, column=2, padx=10, pady=5)

        self.btn_close = ctk.CTkButton(self.sidebar, text="閉じる (👌)", command=self.close_window)
        self.btn_close.grid(row=0, column=3, padx=10, pady=5)
        
        self.btn_history = ctk.CTkButton(self.sidebar, text="履歴を見る", command=self.load_history, fg_color="#555555")
        self.btn_history.grid(row=0, column=4, padx=10, pady=5)

        self.btn_back = ctk.CTkButton(self.sidebar, text="戻る", command=self.restore_original_text, fg_color="#777777")
        self.btn_back.grid(row=0, column=5, padx=10, pady=5)
        self.btn_back.grid_remove() # Hidden initially

        self.btn_more = ctk.CTkButton(self.sidebar, text="さらに詳しく (✌️)", command=self.generate_html_report)
        self.btn_more.grid(row=0, column=6, padx=10, pady=10)

        # Main Content
        self.main_area = ctk.CTkFrame(self)
        self.main_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(self.main_area, font=("Meiryo", 14))
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.history_frame = ctk.CTkScrollableFrame(self.main_area, label_text="履歴一覧")
        self.history_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.history_frame.grid_remove() # Hidden initially

        self.input_frame = ctk.CTkFrame(self.main_area)
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="追加の質問...")
        self.entry.pack(side="left", fill="x", expand=True, padx=5)
        
        self.send_btn = ctk.CTkButton(self.input_frame, text="送信", width=60, command=self.send_followup)
        self.send_btn.pack(side="right", padx=5)

        self.camera_active = False
        self.cap = None
        self.camera_thread = None
        self.camera_active = False
        self.cap = None
        self.camera_thread = None
        self.stop_camera_event = threading.Event()
        
        self.original_text = None

    def load_history(self):
        # Save current text if not already viewing history
        if self.original_text is None:
            self.original_text = self.textbox.get("0.0", "end")
        
        # Show Back button so user can cancel/return
        self.btn_back.grid()

        # Switch view
        self.textbox.grid_remove()
        self.history_frame.grid()
        
        # Clear previous items
        for widget in self.history_frame.winfo_children():
            widget.destroy()

        # List files
        files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
        files.sort(reverse=True) # Newest first

        if not files:
            ctk.CTkLabel(self.history_frame, text="履歴はありません").pack(pady=10)
            return

        for filename in files:
            btn = ctk.CTkButton(
                self.history_frame, 
                text=filename, 
                command=lambda f=filename: self.open_history_file(f),
                fg_color="transparent",
                border_width=1,
                text_color=("gray10", "gray90")
            )
            btn.pack(fill="x", padx=5, pady=2)

    def open_history_file(self, filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.set_text(content)
            
            # Switch back to text view
            self.history_frame.grid_remove()
            self.textbox.grid()
            
            # Ensure Back button is visible (it should be already)
            self.btn_back.grid()
            # Ensure History button is visible (user request)
            self.btn_history.grid()
            
        except Exception as e:
            print(f"Error loading file: {e}")

    def restore_original_text(self):
        # Switch back to text view if in history list
        self.history_frame.grid_remove()
        self.textbox.grid()

        if self.original_text is not None:
            self.set_text(self.original_text)
            self.original_text = None
            self.btn_back.grid_remove()
            # History button remains visible
            self.btn_history.grid()

    def set_text(self, text):
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", text)

    def append_text(self, text):
        self.textbox.insert("end", "\n\n" + text)
        self.textbox.see("end")

    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera_active: return
        self.camera_active = True
        self.cam_btn.configure(text="📷 OFF", fg_color="red")
        self.stop_camera_event.clear()
        self.cap = cv2.VideoCapture(0)
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_camera(self):
        if not self.camera_active: return
        self.camera_active = False
        self.cam_btn.configure(text="📷 ON", fg_color="green")
        self.stop_camera_event.set()
        if self.cap:
            self.cap.release()
        if self.cap:
            self.cap.release()
        # self.cam_label.configure(image=None, text="Camera OFF")

    def camera_loop(self):
        recognizer = GestureRecognizer()
        last_gesture_time = 0
        
        while not self.stop_camera_event.is_set() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process Gesture
            gesture, landmarks = recognizer.process_frame(frame)
            
            # Draw landmarks
            if landmarks:
                recognizer.mp_draw.draw_landmarks(frame, landmarks, recognizer.mp_hands.HAND_CONNECTIONS)

            # Handle Gestures (with debounce)
            current_time = time.time()
            if gesture and (current_time - last_gesture_time > 1.5):
                if gesture == "GOOD":
                    self.after(0, self.save_and_close)
                    last_gesture_time = current_time
                elif gesture == "OK": # Using OK for "Close without saving" as per prompt logic mapping (though prompt said OK is close)
                    self.after(0, self.close_window)
                    last_gesture_time = current_time
                elif gesture == "PEACE":
                    self.after(0, self.generate_html_report)
                    last_gesture_time = current_time
                elif gesture == "PINCH":
                    # Scroll logic could be continuous, but let's just page down for simplicity or step scroll
                    self.after(0, lambda: self.textbox.yview_scroll(1, "units"))
                    # No debounce for scroll or small debounce
                    last_gesture_time = current_time - 1.4 # Allow faster scrolling

            # Update GUI
            # Update GUI - Camera Monitor Removed
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (240, 180))
            # img = Image.fromarray(frame)
            # imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(240, 180))
            # self.after(0, lambda i=imgtk: self.cam_label.configure(image=i, text=""))
            
            time.sleep(0.03)

        if self.cap:
            self.cap.release()

    def save_and_close(self):
        text = self.textbox.get("0.0", "end")
        filename = f"note_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        self.on_close()

    def close_window(self):
        self.on_close()

    def on_close(self):
        self.stop_camera()
        self.withdraw()
        self.controller.on_window_close()

    def generate_html_report(self):
        text = self.textbox.get("0.0", "end")
        threading.Thread(target=self._fetch_html_report, args=(text,), daemon=True).start()

    def _fetch_html_report(self, text):
        html_content = self.controller.gemini.generate_html_report(text)
        filename = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        webbrowser.open(f"file:///{filepath}")

    def send_followup(self):
        query = self.entry.get()
        if not query: return
        self.entry.delete(0, "end")
        self.append_text(f"Q: {query}")
        
        # Run in thread
        threading.Thread(target=self._fetch_followup, args=(query,), daemon=True).start()

    def _fetch_followup(self, query):
        response = self.controller.gemini.explain_text(query)
        self.after(0, lambda: self.append_text(f"A: {response}"))


# --- Main Controller ---

class AppController:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.withdraw() # Hide root window
        
        self.gemini = GeminiClient(GENAI_API_KEY)
        self.overlay = OverlayButton(
            self.on_settings_click,
            self.on_search_hover_start,
            self.on_search_hover_end
        )
        self.quick_info = QuickInfoWindow()
        self.result_window = None
        
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.mouse_listener.start()
        
        self.keyboard_listener = keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+q': self.quit_app
        })
        self.keyboard_listener.start()

        self.drag_start_pos = None
        self.is_dragging = False
        self.search_hovering = False

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            if pressed:
                self.drag_start_pos = (x, y)
                self.is_dragging = True
                
                # Check if we clicked outside the overlay to hide it
                if self.overlay.state() == "normal" and not self.overlay.is_inside(x, y):
                    self.root.after(0, self.overlay.withdraw)
            else:
                if self.is_dragging and self.drag_start_pos:
                    x_diff = abs(x - self.drag_start_pos[0])
                    y_diff = abs(y - self.drag_start_pos[1])
                    if x_diff > DRAG_THRESHOLD and y_diff < 30:
                        # Show overlay
                        self.root.after(0, lambda: self.overlay.show_at(x, y))
                    else:
                        # Hide overlay if clicked elsewhere (and not on the button itself)
                        # Note: This logic is tricky because clicking the button also triggers this.
                        # However, the button is a separate window, so clicks on it might not register here 
                        # or we need to check bounds. 
                        # For simplicity, we hide it on any click that isn't a drag.
                        # But if we click the button, we don't want to hide it immediately before the command fires.
                        # The button command handles the hide.
                        # We can add a small delay or check if the click was on the overlay window (hard from here).
                        # Let's just rely on the fact that if they click elsewhere, it hides.
                        # If they click the button, the button callback fires.
                        pass
                        # Actually, if the user clicks *elsewhere* it should hide.
                        # If the user clicks the button, this listener might fire too.
                        # We'll let the button handle its own click.
                        # We can hide it if the click is far from the button?
                        # Let's just hide it if it's visible and the click is not a drag.
                        if self.overlay.state() == "normal":
                             # Check if click is inside button geometry?
                             # Simplified: Just hide it. If they clicked the button, the button event queue should handle it?
                             # Tkinter events vs Pynput events. Pynput sees it first usually.
                             # If we hide it here, the button might not get the click.
                             # Let's NOT hide it here immediately. Let's use a timer or check focus.
                             pass
                    
                self.is_dragging = False
                self.drag_start_pos = None

    def on_settings_click(self):
        self.copy_selection()
        text = pyperclip.paste()
        
        # Open Result Window
        if not self.result_window or not self.result_window.winfo_exists():
            self.result_window = ResultWindow(self)
        
        self.result_window.deiconify()
        self.result_window.lift()
        self.result_window.set_text("Loading explanation...")
        self.result_window.start_camera() # Auto start camera
        
        # Call API
        threading.Thread(target=self.fetch_explanation, args=(text,), daemon=True).start()

    def on_search_hover_start(self):
        self.search_hovering = True
        self.copy_selection()
        text = pyperclip.paste()
        
        # Show Loading
        self.quick_info.show_text("Loading...", self.overlay.x, self.overlay.y + 50)
        
        # Fetch Quick Info
        threading.Thread(target=self.fetch_quick_info, args=(text,), daemon=True).start()

    def on_search_hover_end(self):
        self.search_hovering = False
        # Hide after a short delay or immediately? 
        # User said "Hover to show". Usually hide on leave.
        self.root.after(100, self.quick_info.hide)

    def fetch_quick_info(self, text):
        info = self.gemini.quick_explain(text)
        self.root.after(0, lambda: self._show_quick_info_if_hovering(info))

    def _show_quick_info_if_hovering(self, info):
        if self.search_hovering:
            self.quick_info.show_text(info, self.overlay.x, self.overlay.y + 50)

    def copy_selection(self):
        # Simulate Ctrl+C
        k = keyboard.Controller()
        with k.pressed(keyboard.Key.ctrl):
            k.press('c')
            k.release('c')
        time.sleep(0.1)

    def fetch_explanation(self, text):
        explanation = self.gemini.explain_text(text)
        self.root.after(0, lambda: self.result_window.set_text(explanation))

    def on_window_close(self):
        # Called when result window is closed
        pass

    def quit_app(self):
        print("Quitting...")
        if self.result_window:
            self.result_window.stop_camera()
        self.root.quit()
        sys.exit()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AppController()
    app.run()
