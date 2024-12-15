import cv2
import face_recognition
import sqlite3
import numpy as np
import os
import requests
from datetime import datetime, timedelta

class FaceRecognitionSystem:
    def __init__(self, database_path='face_database.db', line_token=None):
        """
        เริ่มต้นระบบจดจำใบหน้า
        - สร้างฐานข้อมูล SQLite 
        - เตรียมการส่งแจ้งเตือน Line
        """
        # เชื่อมต่อและสร้างฐานข้อมูล
        self.conn = sqlite3.connect(database_path)
        self.cursor = self.conn.cursor()
        
        # สร้างตาราง users หากยังไม่มี
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                user_code TEXT NOT NULL,
                face_encoding BLOB NOT NULL
            )
        ''')
        
        # สร้างตารางบันทึกการแจ้งเตือน
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                user_id INTEGER,
                last_notification DATETIME,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        self.conn.commit()

        # Line Notify Token
        self.line_token = line_token

    def send_line_notification(self, name, user_code, frame):
        """
        ส่งการแจ้งเตือนผ่าน Line
        - ตรวจสอบเวลาการแจ้งเตือนล่าสุด
        - ส่งรูปภาพและข้อมูลผ่าน Line Notify
        """
        if not self.line_token:
            print("ไม่มี Line Token")
            return False

        try:
            # ตรวจสอบการแจ้งเตือนล่าสุด
            self.cursor.execute('''
                SELECT last_notification FROM notifications 
                JOIN users ON notifications.user_id = users.id 
                WHERE users.name = ?
            ''', (name,))
            
            result = self.cursor.fetchone()
            current_time = datetime.now()

            # ตรวจสอบเวลาการแจ้งเตือน
            if result and (current_time - datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')) < timedelta(minutes=5):
                print(f"แจ้งเตือนล่าสุดของ {name} ยังไม่ครบ 5 นาที")
                return False

            # บันทึกรูปภาพชั่วคราว
            filename = f"detected_face_{name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            # เตรียมส่งไฟล์
            url = "https://notify-api.line.me/api/notify"
            headers = {
                "Authorization": f"Bearer {self.line_token}"
            }
            
            payload = {
                "message": f"ตรวจพบ: {name} (รหัสผู้ใช้: {user_code})"
            }
            
            with open(filename, "rb") as image_file:
                files = {"imageFile": image_file}
                response = requests.post(url, headers=headers, data=payload, files=files)

            # ลบไฟล์ชั่วคราว
            os.remove(filename)

            # บันทึกเวลาการแจ้งเตือน
            self.cursor.execute('''
                INSERT OR REPLACE INTO notifications (user_id, last_notification)
                SELECT id, ? FROM users WHERE name = ?
            ''', (current_time.strftime('%Y-%m-%d %H:%M:%S'), name))
            self.conn.commit()

            print(f"ส่งการแจ้งเตือนสำหรับ {name} สำเร็จ")
            return True

        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการส่งการแจ้งเตือน: {e}")
            return False

    def add_new_user(self, name, user_code, face_image):
        """
        เพิ่มผู้ใช้ใหม่เข้าสู่ระบบ
        """
        face_locations = face_recognition.face_locations(face_image)
        face_encodings = face_recognition.face_encodings(face_image, face_locations)
        
        if face_encodings:
            face_encoding_bytes = face_encodings[0].tobytes()
            
            self.cursor.execute(
                'INSERT INTO users (name, user_code, face_encoding) VALUES (?, ?, ?)', 
                (name, user_code, face_encoding_bytes)
            )
            self.conn.commit()
            print(f"เพิ่มผู้ใช้ {name} สำเร็จ")
            return True
        return False

    def recognize_faces(self):
        """
        ระบบตรวจจับและจดจำใบหน้าแบบเรียลไทม์
        """
        video_capture = cv2.VideoCapture(1)
        
        while True:
            ret, frame = video_capture.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ตรวจจับตำแหน่งใบหน้า
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # แปลงค่ากลับเป็นสี BGR สำหรับ OpenCV
                    top *= 1
                    right *= 1
                    bottom *= 1
                    left *= 1

                    # ค้นหาใบหน้าที่ตรงกันในฐานข้อมูล
                    matches = self.compare_face_with_database(face_encoding)
                    
                    name = "Unknown"
                    user_code = ""
                    
                    if matches:
                        name, user_code = matches
                        print(f"ตรวจพบ: {name} (รหัสผู้ใช้: {user_code})")
                        
                        # ส่งการแจ้งเตือนผ่าน Line
                        self.send_line_notification(name, user_code, frame)
                    
                    # วาดกรอบใบหน้า
                    color = (0, 255, 0)  # สีเขียว
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # เพิ่มข้อความชื่อและรหัสผู้ใช้
                    label = f"{name} ({user_code})"
                    cv2.putText(frame, label, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # แสดงภาพจากกล้อง
            cv2.imshow('Face Recognition', frame)
            
            # กด 'q' เพื่อออก
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # ปิดการเชื่อมต่อ
        video_capture.release()
        cv2.destroyAllWindows()

    def compare_face_with_database(self, face_encoding):
        """
        เปรียบเทียบใบหน้ากับฐานข้อมูล
        """
        self.cursor.execute('SELECT name, user_code, face_encoding FROM users')
        database_faces = self.cursor.fetchall()
        
        for name, user_code, stored_encoding_bytes in database_faces:
            stored_encoding = np.frombuffer(stored_encoding_bytes)
            match = face_recognition.compare_faces([stored_encoding], face_encoding)
            
            if match[0]:
                return name, user_code
        
        return None

    def close_connection(self):
        """
        ปิดการเชื่อมต่อฐานข้อมูล
        """
        self.conn.close()

def main():
    # สร้างระบบโดยใส่ Line Token
    line_token = "Line-token"  # แทนที่ด้วย Token จริง
    face_system = FaceRecognitionSystem(line_token=line_token)

    # Add user
    #person1 = face_recognition.load_image_file("./person1.jpg")
    #face_system.add_new_user("NAME", "ID123", person1)

    try:
        face_system.recognize_faces()
    finally:
        face_system.close_connection()
    
if __name__ == "__main__":
    main()
