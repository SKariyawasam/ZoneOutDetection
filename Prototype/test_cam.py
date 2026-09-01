import cv2

for idx in [0, 1, 2]:
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(idx, backend)
        opened = cap.isOpened()
        if opened:
            ret, frame = cap.read()
            print(f"Index {idx}, Backend {backend}: Opened={opened}, Read={ret}, Shape={frame.shape if ret else None}")
            cap.release()
        else:
            print(f"Index {idx}, Backend {backend}: Failed to open")
