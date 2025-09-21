"""
Simple Face Recognition Attendance System
All-in-one file for maximum simplicity and better accuracy
"""

import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import sqlite3
from datetime import datetime, timedelta
import json
from PIL import Image, ImageTk
import threading
import time
import shutil

class SimpleFaceRecognition:
    def __init__(self):
        # Initialize paths
        self.data_dir = "data"
        self.faces_dir = os.path.join(self.data_dir, "faces")
        self.db_path = os.path.join(self.data_dir, "attendance.db")
        
        # Create directories
        os.makedirs(self.faces_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Recognition parameters - improved for better accuracy
        self.recognition_threshold = 0.8  # Lower threshold for easier recognition
        self.min_face_size = (30, 30)
        
        # Known faces storage
        self.known_faces = {}  # {name: [face_encodings]}
        self.load_known_faces()
        
        # Initialize database
        self.init_database()
        
        # Camera
        self.camera = None
        self.is_recognizing = False
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL
            )
        ''')
        conn.commit()
        conn.close()
    
    def extract_face_features(self, face_img):
        """Extract improved face features for better recognition accuracy"""
        try:
            # Resize to standard size
            face_img = cv2.resize(face_img, (128, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            # Simple but effective feature extraction
            features = []
            
            # 1. Raw pixel values (normalized)
            resized_gray = cv2.resize(gray, (32, 32))
            features.extend(resized_gray.flatten() / 255.0)
            
            # 2. Histogram features
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            features.extend(hist.flatten() / np.sum(hist))
            
            # 3. Simple gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [32], [0, 256])
            features.extend(grad_hist.flatten() / (np.sum(grad_hist) + 1e-7))
            
            # Convert to numpy array and normalize
            features = np.array(features, dtype=np.float32)
            
            # Normalize the entire feature vector
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.array([])
    

    
    def add_person(self, name, images):
        """Add a person with their face images"""
        person_dir = os.path.join(self.faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        
        face_encodings = []
        
        for i, img in enumerate(images):
            # Save image
            img_path = os.path.join(person_dir, f"{name}_{i+1}.jpg")
            cv2.imwrite(img_path, img)
            
            # Extract features
            features = self.extract_face_features(img)
            if len(features) > 0:
                face_encodings.append(features)
        
        if face_encodings:
            self.known_faces[name] = face_encodings
            self.save_face_data()
            return True
        return False
    
    def save_face_data(self):
        """Save face data to disk"""
        face_data = {}
        for name, encodings in self.known_faces.items():
            face_data[name] = [encoding.tolist() for encoding in encodings]
        
        with open(os.path.join(self.data_dir, "face_data.json"), "w") as f:
            json.dump(face_data, f)
    
    def load_known_faces(self):
        """Load known faces from disk"""
        face_data_path = os.path.join(self.data_dir, "face_data.json")
        if os.path.exists(face_data_path):
            try:
                with open(face_data_path, "r") as f:
                    face_data = json.load(f)
                
                # Check for compatibility - if feature sizes don't match, clear old data
                for name, encodings in face_data.items():
                    if encodings and len(encodings[0]) != 1120:  # Expected feature size
                        print(f"Feature size mismatch detected. Clearing old face data for compatibility.")
                        self.known_faces = {}
                        # Remove the old file
                        os.remove(face_data_path)
                        return
                
                # Load compatible data
                for name, encodings in face_data.items():
                    self.known_faces[name] = [np.array(encoding, dtype=np.float32) for encoding in encodings]
                    
                print(f"Loaded {len(self.known_faces)} people from saved data")
                    
            except Exception as e:
                print(f"Error loading face data: {e}")
                self.known_faces = {}
        else:
            print("No existing face data found")
    
    def recognize_face(self, face_img):
        """Recognize a face with improved accuracy"""
        if not self.known_faces:
            return "Unknown", 0.0
        
        # Extract features from the input face
        face_features = self.extract_face_features(face_img)
        if len(face_features) == 0:
            return "Unknown", 0.0
        
        best_match = "Unknown"
        best_confidence = 0.0
        
        # Compare with all known faces
        for name, known_encodings in self.known_faces.items():
            confidences = []
            
            for known_encoding in known_encodings:
                # Calculate cosine similarity
                dot_product = np.dot(face_features, known_encoding)
                norm_a = np.linalg.norm(face_features)
                norm_b = np.linalg.norm(known_encoding)
                
                if norm_a > 0 and norm_b > 0:
                    similarity = dot_product / (norm_a * norm_b)
                    # Convert to percentage-like confidence
                    confidence = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
                    confidences.append(confidence)
            
            # Take the best match for this person
            if confidences:
                max_confidence = max(confidences)
                
                if max_confidence > best_confidence:
                    best_confidence = max_confidence
                    best_match = name
        
        # Apply threshold (lower threshold for easier recognition)
        if best_confidence < 0.5:  # Lower threshold
            return "Unknown", best_confidence
        
        return best_match, best_confidence
    
    def detect_faces(self, frame):
        """Detect faces in frame with improved parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improved face detection parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,     # Smaller scale factor for better detection
            minNeighbors=3,       # Reduced neighbors for more detections
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def log_attendance(self, name, confidence):
        """Log attendance to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if already logged in the last 5 minutes to avoid duplicates
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            cursor.execute('''
                SELECT COUNT(*) FROM attendance 
                WHERE name = ? AND timestamp > ?
            ''', (name, five_minutes_ago))
            
            if cursor.fetchone()[0] == 0:  # Not logged in last 5 minutes
                cursor.execute('''
                    INSERT INTO attendance (name, confidence) 
                    VALUES (?, ?)
                ''', (name, confidence))
                conn.commit()
                print(f"Attendance logged: {name} ({confidence:.2f})")
                conn.close()
                return True
            
            conn.close()
            return False
            
        except Exception as e:
            print(f"Error logging attendance: {e}")
            return False
    
    def get_attendance_records(self):
        """Get all attendance records"""
        try:
            conn = sqlite3.connect(self.db_path)
            # Use direct SQL query instead of pandas to avoid issues
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
            records = cursor.fetchall()
            
            # Get column names
            cursor.execute("PRAGMA table_info(attendance)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            conn.close()
            
            # Convert to pandas DataFrame
            if records:
                df = pd.DataFrame(records, columns=column_names)
                
                # Fix confidence column if it's stored as binary
                if 'confidence' in df.columns:
                    # Convert binary confidence values to float
                    def fix_confidence(val):
                        if isinstance(val, bytes):
                            # Convert binary to float
                            import struct
                            try:
                                return struct.unpack('f', val)[0]
                            except:
                                return 0.0
                        elif isinstance(val, (int, float)):
                            return float(val)
                        else:
                            return 0.0
                    
                    df['confidence'] = df['confidence'].apply(fix_confidence)
                
                print(f"Retrieved {len(df)} attendance records")
                return df
            else:
                print("No attendance records found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting attendance records: {e}")
            return pd.DataFrame()
    
    def export_attendance(self, filename):
        """Export attendance to CSV"""
        df = self.get_attendance_records()
        if not df.empty:
            df.to_csv(filename, index=False)
            return True
        return False


class FaceRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Face Recognition Attendance")
        self.root.geometry("800x600")
        
        # Initialize face recognition system
        self.face_system = SimpleFaceRecognition()
        
        # GUI variables
        self.video_label = None
        self.is_running = False
        self.camera = None
        
        self.create_gui()
        
    def create_gui(self):
        """Create the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="5")
        video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Camera Off", background="black", foreground="white")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Control buttons frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        ttk.Button(control_frame, text="Add Person", command=self.add_person).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Start Recognition", command=self.start_recognition).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Stop Recognition", command=self.stop_recognition).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Import Images", command=self.import_images).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Export Attendance", command=self.export_attendance).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="View Records", command=self.view_records).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Clear Data", command=self.clear_all_data).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Test Database", command=self.test_database).grid(row=2, column=1, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.status_text = tk.Text(status_frame, height=10, width=30)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self.log_status("System initialized. Add people and start recognition.")
    
    def log_status(self, message):
        """Log message to status window"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\\n")
        self.status_text.see(tk.END)
    
    def add_person(self):
        """Add a new person using camera"""
        name = simpledialog.askstring("Add Person", "Enter person's name:")
        if not name:
            return
        
        name = name.strip().replace(' ', '_')
        
        # Open camera for capturing
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        
        self.log_status(f"Adding person: {name}. Press SPACE to capture, ESC to finish.")
        
        captured_faces = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = self.face_system.detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Show instructions
            cv2.putText(frame, f"Capturing for: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {len(captured_faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture, ESC: Finish", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Add Person', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to capture
                if len(faces) > 0:
                    x, y, w, h = faces[0]  # Take the first face
                    face_img = frame[y:y+h, x:x+w]
                    captured_faces.append(face_img)
                    self.log_status(f"Captured image {len(captured_faces)} for {name}")
            elif key == 27:  # ESC to finish
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_faces:
            success = self.face_system.add_person(name, captured_faces)
            if success:
                self.log_status(f"Successfully added {name} with {len(captured_faces)} images")
                self.log_status(f"Total people in system: {len(self.face_system.known_faces)}")
                messagebox.showinfo("Success", f"Added {name} successfully!")
            else:
                self.log_status(f"Failed to add {name}")
                messagebox.showerror("Error", "Failed to add person")
        else:
            self.log_status("No faces captured")
    
    def import_images(self):
        """Import images for a person"""
        name = simpledialog.askstring("Import Images", "Enter person's name:")
        if not name:
            return
        
        name = name.strip().replace(' ', '_')
        
        files = filedialog.askopenfilenames(
            title="Select face images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not files:
            return
        
        imported_faces = []
        for file_path in files:
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    # Try to detect face in the image
                    faces = self.face_system.detect_faces(img)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Take the first face
                        face_img = img[y:y+h, x:x+w]
                        imported_faces.append(face_img)
                    else:
                        # If no face detected, use the whole image (might be a cropped face)
                        imported_faces.append(img)
            except Exception as e:
                self.log_status(f"Error importing {file_path}: {e}")
        
        if imported_faces:
            success = self.face_system.add_person(name, imported_faces)
            if success:
                self.log_status(f"Successfully imported {len(imported_faces)} images for {name}")
                self.log_status(f"Total people in system: {len(self.face_system.known_faces)}")
                messagebox.showinfo("Success", f"Imported {len(imported_faces)} images for {name}")
            else:
                messagebox.showerror("Error", "Failed to import images")
        else:
            messagebox.showwarning("Warning", "No valid face images found")
    
    def start_recognition(self):
        """Start face recognition"""
        if not self.face_system.known_faces:
            messagebox.showwarning("Warning", "No people added yet. Add people first.")
            return
        
        if self.is_running:
            return
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
        
        self.is_running = True
        self.log_status("Starting face recognition...")
        self.recognition_loop()
    
    def recognition_loop(self):
        """Main recognition loop"""
        if not self.is_running:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # Detect faces
            faces = self.face_system.detect_faces(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                name, confidence = self.face_system.recognize_face(face_img)
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Log attendance if recognized
                if name != "Unknown" and confidence > 0.5:  # Use the same threshold as in recognize_face
                    if self.face_system.log_attendance(name, confidence):
                        self.log_status(f"Attendance: {name} ({confidence:.2f})")
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_pil = frame_pil.resize((400, 300))
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
        
        # Schedule next frame
        self.root.after(33, self.recognition_loop)  # ~30 FPS
    
    def stop_recognition(self):
        """Stop face recognition"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.video_label.configure(image="", text="Camera Off")
        self.log_status("Stopped face recognition")
    
    def export_attendance(self):
        """Export attendance records"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if filename:
            if self.face_system.export_attendance(filename):
                self.log_status(f"Exported attendance to {filename}")
                messagebox.showinfo("Success", f"Exported to {filename}")
            else:
                messagebox.showerror("Error", "Failed to export attendance")
    
    def view_records(self):
        """View attendance records"""
        try:
            # Add debugging
            self.log_status("Loading attendance records...")
            df = self.face_system.get_attendance_records()
            
            if df.empty:
                # Check if there are any people added
                if self.face_system.known_faces:
                    people_list = ", ".join(self.face_system.known_faces.keys())
                    # Check database directly
                    try:
                        conn = sqlite3.connect(self.face_system.db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM attendance")
                        db_count = cursor.fetchone()[0]
                        conn.close()
                        
                        message = f"Database contains {db_count} records but failed to load.\n\n"
                        message += f"People in system: {people_list}\n\n"
                        message += "Try starting recognition and being detected by the camera."
                        
                    except Exception as e:
                        message = f"Database error: {e}\n\n"
                        message += f"People in system: {people_list}\n\n"
                        message += "Try restarting the application."
                    
                    messagebox.showinfo("Attendance Records Issue", message)
                else:
                    messagebox.showinfo("No Data", "No people added yet. Add people first, then start recognition.")
                return
            
            self.log_status(f"Found {len(df)} attendance records")
            
            # Create a new window to show records
            records_window = tk.Toplevel(self.root)
            records_window.title(f"Attendance Records ({len(df)} total)")
            records_window.geometry("700x400")
            
            # Create treeview
            tree = ttk.Treeview(records_window, columns=("ID", "Name", "Timestamp", "Confidence"), show="headings")
            tree.heading("ID", text="ID")
            tree.heading("Name", text="Name")
            tree.heading("Timestamp", text="Timestamp")
            tree.heading("Confidence", text="Confidence")
            
            # Set column widths
            tree.column("ID", width=50)
            tree.column("Name", width=150)
            tree.column("Timestamp", width=200)
            tree.column("Confidence", width=100)
            
            # Insert data
            for _, row in df.iterrows():
                tree.insert("", "end", values=(row['id'], row['name'], row['timestamp'], f"{row['confidence']:.2f}"))
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(records_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            scrollbar.pack(side="right", fill="y", pady=10)
            
            # Add summary label
            summary_label = ttk.Label(records_window, text=f"Total Records: {len(df)}")
            summary_label.pack(pady=5)
            
        except Exception as e:
            self.log_status(f"Error viewing records: {e}")
            messagebox.showerror("Error", f"Failed to load records: {e}")
    
    def clear_all_data(self):
        """Clear all face data and start fresh"""
        result = messagebox.askyesno(
            "Clear All Data", 
            "This will delete all people and attendance records.\n\nAre you sure?"
        )
        
        if result:
            try:
                # Clear face system data
                self.face_system.known_faces = {}
                
                # Remove face data file
                face_data_path = os.path.join(self.face_system.data_dir, "face_data.json")
                if os.path.exists(face_data_path):
                    os.remove(face_data_path)
                
                # Clear database
                conn = sqlite3.connect(self.face_system.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance")
                conn.commit()
                conn.close()
                
                # Clear faces directory
                faces_dir = os.path.join(self.face_system.data_dir, "faces")
                if os.path.exists(faces_dir):
                    shutil.rmtree(faces_dir)
                    os.makedirs(faces_dir, exist_ok=True)
                
                self.log_status("All data cleared successfully")
                messagebox.showinfo("Success", "All data has been cleared. You can now start fresh.")
                
            except Exception as e:
                self.log_status(f"Error clearing data: {e}")
                messagebox.showerror("Error", f"Failed to clear data: {e}")
    
    def test_database(self):
        """Test database connection and show debug info"""
        try:
            conn = sqlite3.connect(self.face_system.db_path)
            cursor = conn.cursor()
            
            # Check table structure
            cursor.execute("PRAGMA table_info(attendance)")
            columns = cursor.fetchall()
            
            # Count records
            cursor.execute("SELECT COUNT(*) FROM attendance")
            total_count = cursor.fetchone()[0]
            
            # Get sample records
            cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 5")
            sample_records = cursor.fetchall()
            
            conn.close()
            
            # Show results
            result_text = f"Database Test Results:\n\n"
            result_text += f"Total Records: {total_count}\n\n"
            result_text += f"Table Columns: {[col[1] for col in columns]}\n\n"
            
            if sample_records:
                result_text += "Sample Records:\n"
                for record in sample_records:
                    result_text += f"  {record}\n"
            else:
                result_text += "No records found\n"
            
            result_text += f"\nKnown Faces: {list(self.face_system.known_faces.keys())}"
            
            messagebox.showinfo("Database Test", result_text)
            self.log_status(f"Database test: {total_count} records found")
            
        except Exception as e:
            error_msg = f"Database test failed: {e}"
            self.log_status(error_msg)
            messagebox.showerror("Database Error", error_msg)
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()


def main():
    """Main function"""
    print("Starting Simple Face Recognition Attendance System...")
    app = FaceRecognitionGUI()
    app.run()


if __name__ == "__main__":
    main()