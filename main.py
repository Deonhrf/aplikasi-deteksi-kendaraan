# Meotde deteksi dengan gambar

from ultralytics import YOLO
import os
import shutil  # untk menyalin/upload folder
import cv2 # untuk memproses gambar 

# path folder dan model
train = 'train'
results = 'results'
path_model = 'best.pt'

# load model Yolo
def load_model() :
  model = YOLO(path_model)

  # Folder hasil deteksi
  if os.path.exists(results) :
    shutil.rmtree(results) # Hapus direktori serta isi sebelumnya
    os.makedirs(results) # membuat folder baru

    # Prose gambar
    for path_images in os.listdir(train) :
      input_path = os.path.join(train,path_images)
      output_path = os.path.join(results, path_images)

      if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')) :
        continue

      try :
        result = model.predict(source=input_path, conf = 0.5, save=False)
        for i in result :
          bounding_box = i.plot()
          cv2.imwrite(output_path, bounding_box) # disimpan gambar yang telah dianotasikan ke path output
          print(f'Gambar Berhasil diproses & disimpan di {path_images}')

      except Exception as e :
        print(f'Gambar gagal diproses {path_images} : {e}')

if __name__ == '__main__' :
  load_model()
