from ultralytics import YOLO
import custom_counter
import cv2

model = YOLO("yolov8n.pt")
# Realtime
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Video
cap = cv2.VideoCapture(r"Videotest/cars.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# Define line points
region_points  = [(120, 280), (580, 280), (120, 190), (150, 240)]

# Init Object Counter
counter = custom_counter.ObjectCounter()
counter.set_arguments(view_img = True,
                 reg_pts = region_points,
                 classes_names = model.names,
                 draw_tracks = True
                 )


while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist = True, show = False, conf = 0.5, classes = 2)

    # Load counting region
    im0 = counter.start_counting(im0, tracks)
    # Display the resulting frame
    cv2.imshow("YOLOv8 Object Counter", im0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # video_writer.write(im0)

cap.release()
# video_writer.release()
cv2.destroyAllWindows()