import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def boundary(img,classifier,scaleFactor,minNeighbors,color,text): #ขอบเขตหน้า
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray,scaleFactor,minNeighbors)
    coords = []
    for(x,y,w,h) in features: #สร้างกรอบใบหน้า
        cv2.rectangle(img, (x,y), (x+w,y+h), color,2)
        cv2.putText(img,text, (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    return img

def detect(img, face_cascade): #เรียกตรวจจับใบหน้า
    img = boundary(img, face_cascade, 1.3, 10, (0,255,0), "Face")
    return img

webcam = cv2.VideoCapture(0)
while (True):
    check , frame = webcam.read() 
    frame = detect(frame,face_cascade)
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == ord("e"): #กด e เพื่อปิด
        break
    
webcam.release()
cv2.destroyAllWindows()
