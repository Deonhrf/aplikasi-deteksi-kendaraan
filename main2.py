# Metode deteksi secara real-time dengan webcam

from ultralytics import YOLO
import cv2


def main() :
  # load model 
  model = YOLO('best.pt') 

  # Open kamera
  cam = cv2.VideoCapture(0)
  if not cam.isOpened():
    print(f'Kamera tidak dapat dibuka')
    return 
  
  while True :
    ret, frame = cam.read()
    if not ret :
      print('Tidak dapat membaca Frame')
      break

    # Prose prediksi 
    results = model.predict(source=frame, conf=0.4, save=False) # proses prediction
    bounding_box = results[0].plot() # bounding box 
    cv2.imshow('Yolov8 Detection', bounding_box)

    # Keluar dari webcam
    if cv2.waitKey(1) & 0xFF == ord('q') :
      break
  cam.release()
  cv2.destroyAllWindows()
  print('Proses Selesai')

if __name__ == "__main__" :
  main()