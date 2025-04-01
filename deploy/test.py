import speech_recognition as sr

# Khởi tạo recognizer
recognizer = sr.Recognizer()

# Mở micro để thu âm
with sr.Microphone() as source:
    print("Nói gì đó...")
    recognizer.adjust_for_ambient_noise(source)  # Điều chỉnh theo tiếng ồn môi trường
    audio = recognizer.listen(source)  # Lắng nghe giọng nói

# Chuyển giọng nói thành văn bản
try:
    text = recognizer.recognize_google(audio, language="vi-VN")  # Nhận diện tiếng Việt
    print("Bạn vừa nói:", text)
except sr.UnknownValueError:
    print("Không nhận diện được giọng nói!")
except sr.RequestError:
    print("Lỗi khi kết nối tới dịch vụ nhận diện giọng nói!")
# # Lưu file âm thanh (tùy chọn)
# with open("recorded.wav", "wb") as file:
#     file.write(audio.get_wav_data())

# print("Đã thu âm xong và lưu file recorded.wav")
