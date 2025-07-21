# First-Using-YOLO
간단한 Python 프로젝트입니다. 영상(plain.mp4)에 등장하는 객체를 YOLOv8모델을 이용해 탐지하고 시각화합니다.

# Requirements

  pip install ultralytics opencv-python

#How to Run

  python test.py

1. plain.mp4 영상을 읽음
2. YOLOv8모델 로드
3. 각 프레임에서 객체 탐지 수행
4. 탐지 결과를 시각화하여 실시간으로 화면에 출력
5. 'q'를 누르면 종료

#Files
- test.py: 메인 실행 코드
- plane.mp4: 탐지 대상 영상
- yolov8.pt: 사전 학습된 YOLOv8 가중치 파일

#Notes
- 프레임을 읽지 못할 경우 "프레임 읽기 실패!"출력 후 종료됩니다.
- 'q'를 누르면 영상이 종료됩니다.
