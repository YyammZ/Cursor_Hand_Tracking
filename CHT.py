import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
from pynput.keyboard import Listener, Key
import math
import time
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os
import sys
import pyautogui

def get_model_path():
    if getattr(sys, 'frozen', False):  # Jika dalam bentuk .exe
        base_path = sys._MEIPASS  # Temp folder untuk PyInstaller
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))  # Saat dijalankan sebagai script

    return os.path.join(base_path, "mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb")

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Kurangi jumlah tangan yang dideteksi (1 tangan lebih ringan)
    min_detection_confidence=0.5,
    model_complexity=0  # Gunakan model yang lebih ringan
)


# Inisialisasi MediaPipe dan Mouse Controller
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mouse = Controller()

# Resolusi layar
screen_width, screen_height = 1920, 1200

# Kalman Filter untuk pergerakan kursor
class KalmanFilter:
    def __init__(self):
        self.state = np.array([0, 0, 0, 0])  # [x, y, vx, vy]
        self.P = np.eye(4) * 1000  # Covariance matrix
        self.F = np.array([[1, 0, 1, 0],  # State transition model
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],  # Observation model
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * 0.1  # Measurement noise
        self.Q = np.eye(4) * 0.01  # Process noise

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state[:2]

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

# Inisialisasi Kalman Filter
kalman_filter = KalmanFilter()

# Variabel pengaturan
alpha = 0.5
movement_threshold = 50
speed_factor = 3.0
buffer_size = 100
cursor_speed_multiplier = 4.0 
interpolasi = 0.3

adaptive_threshold = True  # Gunakan threshold adaptif
# Variabel untuk menyimpan posisi kursor sebelumnya
prev_smooth_x, prev_smooth_y = 0, 0
is_holding = False
contact_start_time = None  # Waktu mulai kontak jari

# Buffer untuk menyimpan posisi sebelumnya
position_buffer = []

# Fungsi untuk menghitung jarak antara dua titik
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fungsi untuk menghitung rata-rata posisi dari buffer
def get_average_position(buffer):
    if not buffer:
        return 0, 0
    avg_x = sum([pos[0] for pos in buffer]) / len(buffer)
    avg_y = sum([pos[1] for pos in buffer]) / len(buffer)
    return int(avg_x), int(avg_y)

detection_mode = "both"  # Default mode
offset_x, offset_y = -4870, -2480  # Default offset

def on_key_press(key):
    global detection_mode, offset_x, offset_y
    try:
        if key == Key.tab and hasattr(key, 'ctrl_l'):
            if detection_mode == "both":
                detection_mode = "right"
            elif detection_mode == "right":
                detection_mode = "left"
            else:
                detection_mode = "both"
            print(f"Mode changed to: {detection_mode}")
        elif key == Key.up:
            offset_y -= 10
        elif key == Key.down:
            offset_y += 10
        elif key == Key.left:
            offset_x -= 10
        elif key == Key.right:
            offset_x += 10
    except AttributeError:
        pass


# Fungsi utama untuk menjalankan program
def run_hand_tracking():
    global prev_smooth_x, prev_smooth_y, is_holding, contact_start_time, detection_mode

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Atur lebar kamera ke 640px
    cap.set(4, 480)  # Atur tinggi kamera ke 480px
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        prev_right_distance = None
        prev_left_distance = None
        last_zoom_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame dan ubah ke RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            # Proses deteksi tangan
            result = hands.process(rgb_frame)

            # Gambar deteksi tangan
            if result.multi_hand_landmarks and len(result.multi_hand_landmarks) > 1:
                right_hand = result.multi_hand_landmarks[0]
                left_hand = result.multi_hand_landmarks[1]
        
                # Deteksi jari tangan kanan
                right_thumb_tip = right_hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                right_ring_tip = right_hand.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        
                x1_r, y1_r = int(right_thumb_tip.x * frame.shape[1]), int(right_thumb_tip.y * frame.shape[0])
                x2_r, y2_r = int(right_ring_tip.x * frame.shape[1]), int(right_ring_tip.y * frame.shape[0])
        
                right_distance = calculate_distance(x1_r, y1_r, x2_r, y2_r)
        
                # Deteksi jari tangan kiri
                left_thumb_tip = left_hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                left_ring_tip = left_hand.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        
                x1_l, y1_l = int(left_thumb_tip.x * frame.shape[1]), int(left_thumb_tip.y * frame.shape[0])
                x2_l, y2_l = int(left_ring_tip.x * frame.shape[1]), int(left_ring_tip.y * frame.shape[0])
        
                left_distance = calculate_distance(x1_l, y1_l, x2_l, y2_l)
        
                # Zoom in jika ibu jari dan jari manis tangan kanan bersentuhan
                if prev_right_distance is not None and time.time() - last_zoom_time >= 1:
                    if right_distance < 20:  # Threshold jarak untuk mendeteksi sentuhan
                        pyautogui.hotkey('ctrl', '+')  # Zoom in
                        last_zoom_time = time.time()
        
                # Zoom out jika ibu jari dan jari manis tangan kiri bersentuhan
                if prev_left_distance is not None and time.time() - last_zoom_time >= 1:
                    if left_distance < 20:  # Threshold jarak untuk mendeteksi sentuhan
                        pyautogui.hotkey('ctrl', '-')  # Zoom out
                        last_zoom_time = time.time()
        
                prev_right_distance = right_distance
                prev_left_distance = left_distance

            if result.multi_hand_landmarks and result.multi_handedness:
                right_hand_landmarks = None
                left_hand_landmarks = None

                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    hand_label = handedness.classification[0].label

                    # Simpan landmark tangan kanan dan kiri
                    if hand_label == "Right":
                        right_hand_landmarks = hand_landmarks
                    elif hand_label == "Left":
                        left_hand_landmarks = hand_landmarks

                    # Gambar semua tangan yang terdeteksi
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Gunakan tangan kanan atau kiri berdasarkan mode
                active_hand_landmarks = None
                if detection_mode == "right" and right_hand_landmarks:
                    active_hand_landmarks = right_hand_landmarks
                elif detection_mode == "left" and left_hand_landmarks:
                    active_hand_landmarks = left_hand_landmarks
                elif detection_mode == "both" and right_hand_landmarks:
                    active_hand_landmarks = right_hand_landmarks  # Hanya tangan kanan untuk kursor

                if active_hand_landmarks:
                    # Deteksi scrolling
                    if detection_mode == "both":
                        if left_hand_landmarks is not None:
                            # Tangan kanan untuk scroll up
                            right_thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            right_middle_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            if calculate_distance(right_thumb_tip.x, right_thumb_tip.y, right_middle_tip.x, right_middle_tip.y) < 0.02:
                                mouse.scroll(0, -1)  # Scroll up

                            # Tangan kiri untuk scroll down
                        if left_hand_landmarks is not None:
                            left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            left_middle_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            if calculate_distance(left_thumb_tip.x, left_thumb_tip.y, left_middle_tip.x, left_middle_tip.y) < 0.02:
                                mouse.scroll(0, 1)  # Scroll down

                    # Zoom in dengan tangan kanan (ibu jari dan jari manis bersentuhan)
                    if right_hand_landmarks is not None:
                        right_thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        right_ring_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                        if calculate_distance(right_thumb_tip.x, right_thumb_tip.y, right_ring_tip.x, right_ring_tip.y) < 0.02:
                            mouse.scroll(0, 2)  # Zoom in

                    # Zoom out dengan tangan kiri (ibu jari dan jari manis bersentuhan)
                    if left_hand_landmarks is not None:
                        left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        left_ring_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

                        if calculate_distance(left_thumb_tip.x, left_thumb_tip.y, left_ring_tip.x, left_ring_tip.y) < 0.02:
                            mouse.scroll(0, -2)  # Zoom out

                    # Tampilkan mode deteksi di layar
                    mode_text = f"Detection Mode: BOTH"
                    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Variabel global untuk melacak status kontak ibu jari dan telunjuk kiri
                    left_click_active = False  # Apakah klik sedang aktif

                    if detection_mode == "both" and left_hand_landmarks is not None:
                        # Deteksi ujung ibu jari dan ujung telunjuk jari kiri
                        left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        left_index_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        # Hitung jarak antara ibu jari dan telunjuk
                        left_click_distance = calculate_distance(
                            left_thumb_tip.x, left_thumb_tip.y, left_index_tip.x, left_index_tip.y
                        )

                        # Jika jarak lebih kecil dari threshold dan klik belum aktif
                        if left_click_distance < 0.02 and not left_click_active:
                            mouse.click(Button.right, 1)  # Lakukan klik kanan
                            left_click_active = True  # Aktifkan status klik

                        # Reset status klik jika jari tidak menempel
                        elif left_click_distance >= 0.02:
                            left_click_active = False  # Matikan status klik

                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
                        # Deteksi posisi ujung jari tengah
                        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                        raw_x = middle_finger_tip.x * screen_width
                        raw_y = middle_finger_tip.y * screen_height
    
                        # Terapkan Kalman Filter
                        kalman_filter.update(np.array([raw_x, raw_y]))
                        filtered_x, filtered_y = kalman_filter.predict()
    
                        # Terapkan scaling kecepatan dinamis
                        distance_moved = calculate_distance(mouse.position[0], mouse.position[1], filtered_x, filtered_y)
                        dynamic_speed = speed_factor * min(distance_moved / 50.0, 1.0)
    
                        # Perbarui posisi kursor
                        mouse.position = (int(filtered_x * dynamic_speed), int(filtered_y * dynamic_speed))

                    middle_finger_base = active_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    raw_x = middle_finger_base.x * screen_width * cursor_speed_multiplier + offset_x
                    raw_y = middle_finger_base.y * screen_height * cursor_speed_multiplier + offset_y

                    # Buffer untuk menyimpan posisi sebelumnya
                    position_buffer = []
                    
                    # Terapkan Exponential Moving Average (EMA) untuk smoothing
                    smooth_x = int(alpha * raw_x + (1 - alpha) * prev_smooth_x)
                    smooth_y = int(alpha * raw_y + (1 - alpha) * prev_smooth_y)
                    
                    # Tambahkan posisi smoothing saat ini ke buffer
                    position_buffer.append((smooth_x, smooth_y))
                    if len(position_buffer) > buffer_size:
                        position_buffer.pop(0)  # Hapus posisi lama jika buffer penuh
                    
                    # Hitung rata-rata dari buffer untuk stabilisasi
                    avg_x, avg_y = get_average_position(position_buffer)
                    
                    # Interpolasi linier untuk pergerakan halus
                    interp_x = int(prev_smooth_x + (avg_x - prev_smooth_x) * interpolasi)  # 20% interpolasi
                    interp_y = int(prev_smooth_y + (avg_y - prev_smooth_y) * interpolasi)
                    
                    # Perbarui posisi kursor
                    mouse.position = (interp_x, interp_y)
                    
                    # Simpan posisi saat ini sebagai posisi sebelumnya
                    prev_smooth_x, prev_smooth_y = interp_x, interp_y
                                        
                    # Tambahkan posisi ke buffer
                    position_buffer.append((smooth_x, smooth_y))

                    # Batasi posisi kursor agar tidak keluar layar
                    smooth_x = max(0, min(screen_width, smooth_x))
                    smooth_y = max(0, min(screen_height, smooth_y))
                    # Batasi ukuran buffer (misalnya 5)
                    if len(position_buffer) > 5:
                        position_buffer.pop(0)

                    # Hitung rata-rata dari buffer untuk stabilisasi
                    avg_x, avg_y = get_average_position(position_buffer)

                    # Periksa apakah pergerakan cukup besar (di atas ambang batas)
                    if calculate_distance(prev_smooth_x, prev_smooth_y, avg_x, avg_y) > movement_threshold:
                        mouse.position = (int(avg_x), int(avg_y))

                    # Simpan posisi smoothing saat ini sebagai posisi sebelumnya
                    prev_smooth_x, prev_smooth_y = avg_x, avg_y
    
                    # Deteksi klik dan tahan dengan durasi kontak
                    index_finger_tip = active_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = active_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    distance = calculate_distance(index_finger_tip.x, index_finger_tip.y, thumb_tip.x, thumb_tip.y)

                    if distance < 0.02:  # Jika ibu jari dan telunjuk bersentuhan
                        if contact_start_time is None:  # Mulai menghitung durasi kontakb
                            contact_start_time = time.time()
                        elif time.time() - contact_start_time >= 1.0:  # Jika kontak berlangsung >= 1 detik
                            if not is_holding:
                                mouse.press(Button.left)  # Tahan objek
                                is_holding = True
                    else:  # Jika tidak bersentuhan
                        if contact_start_time is not None and time.time() - contact_start_time < 1.0:
                            mouse.click(Button.left, 1)  # Klik objek
                        contact_start_time = None
                        if is_holding:
                            mouse.release(Button.left)  # Lepaskan objek
                            is_holding = False

                # Deteksi scrolling di mode "right"
                if detection_mode in ["right", "both"] and right_hand_landmarks is not None:
                    right_thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    right_middle_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Deteksi kontak ibu jari dan jari tengah
                    if calculate_distance(right_thumb_tip.x, right_thumb_tip.y, right_middle_tip.x, right_middle_tip.y) < 0.02:
                        current_right_y = right_middle_tip.y * screen_height

                        # Inisialisasi posisi awal jika belum ada
                        if prev_right_y is None:
                            prev_right_y = current_right_y

                        delta = current_right_y - prev_right_y

                        # Tentukan arah scrolling
                        if delta > 10:  # Gerakan ke bawah (scroll ke atas)
                            scroll_direction_right = -1
                        elif delta < -10:  # Gerakan ke atas (scroll ke bawah)
                            scroll_direction_right = 1
                        
                        # Scroll terus sampai jari dilepaskan
                        if scroll_direction_right is not None:
                            mouse.scroll(0, scroll_direction_right)
                    else:
                        # Reset jika kontak hilang
                        prev_right_y = None
                        scroll_direction_right = None

                # Deteksi scrolling di mode "left"
                if detection_mode in ["left", "both"] and left_hand_landmarks is not None:
                    left_thumb_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    left_middle_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    # Deteksi kontak ibu jari dan jari tengah
                    if calculate_distance(left_thumb_tip.x, left_thumb_tip.y, left_middle_tip.x, left_middle_tip.y) < 0.02:
                        current_left_y = left_middle_tip.y * screen_height

                        # Inisialisasi posisi awal jika belum ada
                        if prev_left_y is None:
                            prev_left_y = current_left_y

                        delta = current_left_y - prev_left_y

                        # Tentukan arah scrolling
                        if delta > 10:  # Gerakan ke bawah (scroll ke atas)
                            scroll_direction_left = -1
                        elif delta < -10:  # Gerakan ke atas (scroll ke bawah)
                            scroll_direction_left = 1
                        
                        # Scroll terus sampai jari dilepaskan
                        if scroll_direction_left is not None:
                            mouse.scroll(0, scroll_direction_left)
                    else:
                        # Reset jika kontak hilang
                        prev_left_y = None
                        scroll_direction_left = None

            # Tampilkan mode deteksi di layar
            mode_text = f"Detection Mode: {detection_mode.upper()}"
            cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Tampilkan frame kamera di jendela floating
            resized_frame = cv2.resize(frame, (300, 210))
            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            # Pastikan root masih ada sebelum memperbarui gambar
            try:
                if root.winfo_exists():
                    img_tk = ImageTk.PhotoImage(img, master=root)
                    lbl_video.imgtk = img_tk
                    lbl_video.configure(image=img_tk)
                    root.update()
            except tk.TclError:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    try:
        if root.winfo_exists():
            root.destroy()
    except tk.TclError:
        pass


# Buat jendela GUI dengan tkinter
root = tk.Tk()
root.title("Hand Tracking")
root.geometry("305x215")
root.attributes('-topmost', True)  # Jendela selalu di depan
root.resizable(False, False)

# Label untuk menampilkan video
lbl_video = tk.Label(root)
lbl_video.pack()

# Jalankan Listener untuk tombol keyboard
listener = Listener(on_press=on_key_press)
listener.start()

# Jalankan hand tracking setelah GUI Tkinter siap
root.after(100, run_hand_tracking)
root.mainloop()

listener.stop()



