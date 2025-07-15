import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('plane.mp4')
#yolo모델 불러오기
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        print('프레임 읽기 실패!')
        break
    
    # 프레임에서 모델의 예측 결과를 results에 저장하고, results를 시각화하여 frame에 다시 저장.
    results = model(frame)
    frame = results[0].plot()
    
    cv2.imshow('plane detiction', frame)
    if cv2.waitKey(10)&0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

