import cv2

def main():
    #membuat haar cascade untuk detaksi wajah
    #pastikan file'harrcascade_frontalface_default,xml' ada di direktori yang sama
    #atau berikan path lengkap ke file tersebut,
    face_cascade = cv2.cascadeClassifiler(cv2.data.haarcaascades + 'haarcaac_fromtalface_default.xml')

    #inisialisai kamera (0 adalah ID untuk kamera default)
    #jika anda memiliki beberapa kamera,coba ganti angka 0 dengan 1, 2, dst. 
    cap = cv2.vidiocapture(0)

    # periksa apakah kamera berhasil dibuka 
    if not cap.isopened():
        print("Error: tidak dapat membuka kamera.")
        return

    print("tekan 'q' untuk keluar dari jendela deteksi wajah.")

    while True: 
        #baca frame demi frame dari kamera 
        ret, frame = cap.read()

        #jika frame tidak berasil dibaca, keluar dari loop
        if not ret: 
            print("Error: gagal membaca frame.")
            break  
             
        #ubah frame menjadi grayscale untuk deteksi yang lebih cepat
        # Haar cascades bekerja lebih baik pada gambar grayscale
        gray = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah dalam frame grayscale
        # - scalefactor: seberapa banyak ukuran gambar dikurangi pada setiap skala gambar.
        #                ini mengompensasi fakta bahwa bisa lebih dekat atau lebih jauh .
        # - minNeighbors: beberapa banyak yang harus dimiliki setiap kandidat persegi panjang
        #                untuk mempertahankan kandidatnya. Nilai yang lebih tinggi menghasilkan
        #                 lebih sedikit deteksi palsu tetapi dapat melewatkan beberapa wajah.
        # -minSize: Ukur objek minuman yang dianggap sebagai wajah.
        faces = face_cascade.delecMultiscale(gray, scaleFaktor=1.1, minNeighbors=5, minSize=(30,30))
        for (x, y, w, h)in faces:
            cv2.rectangle(frame, (x, y,), (x + w, y + h), 2) # warna biru,tebal 2

            cv2.imshow('detaksi wajah real-time',frame)
            
            if cv2.waitkey(1) & 0xff == ord('q'):
                break

        cap.release()
        cv2.destroyaallwindows()

    if __name__ == "_main_":
        main()