import cv2
import csv
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# LOAD AI MODELS
# ==========================================

ripeness_model = YOLO("models/tomato_ripeness.pt")
growth_model = YOLO("models/tomato_growth.pt")


# ==========================================
# OPEN VIDEO
# ==========================================

video_path = "videos/tomato_growth.mp4.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Could not open video.")
    exit()

print("Starting GreenVision AI video analysis...")


# ==========================================
# STARTING VALUES
# ==========================================

growth_stage = "Analyzing..."
total_tomatoes = 0
ripe_count = 0
halfripe_count = 0
unripe_count = 0

frame_number = 0
detected_frame = None

# Analyze every 30 frames
ANALYZE_EVERY = 60

# ==========================================
# CSV OUTPUT FILE
# ==========================================

csv_file = "greenvision_results.csv"

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "growth_stage",
        "total_tomatoes",
        "ripe",
        "half_ripe",
        "unripe"
    ])


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1


    # --------------------------------------
    # RUN AI PERIODICALLY
    # --------------------------------------

    if frame_number % ANALYZE_EVERY == 0:

        # TOMATO + RIPENESS DETECTION
        ripeness_results = ripeness_model(
            frame,
            conf=0.75,
            verbose=False
        )

        ripeness_result = ripeness_results[0]

        # Create frame with YOLO detection boxes
        detected_frame = ripeness_result.plot()

        ripe_count = 0
        halfripe_count = 0
        unripe_count = 0

        for box in ripeness_result.boxes:

            class_id = int(box.cls[0])
            class_name = ripeness_model.names[class_id]

            if class_name == "ripe":
                ripe_count += 1

            elif class_name == "half_ripe":
                halfripe_count += 1

            elif class_name == "unripe":
                unripe_count += 1


        total_tomatoes = (
            ripe_count
            + halfripe_count
            + unripe_count
        )


        # ----------------------------------
        # DETERMINE GROWTH STAGE
        # ----------------------------------

        if total_tomatoes > 0:

            growth_stage = "Fruiting / Ripening"

        else:

            growth_results = growth_model(
                frame,
                verbose=False
            )

            growth_result = growth_results[0]

            class_id = growth_result.probs.top1
            growth_stage = growth_model.names[class_id]

            if growth_stage == "Early_Vegetative":
                growth_stage = "Early Vegetative"

            elif growth_stage == "Flowering_Initiation":
                growth_stage = "Flowering Initiation"


        # ----------------------------------
        # PRINT RESULTS
        # ----------------------------------

        print()
        print("==============================")
        print("GREENVISION LIVE RESULTS")
        print("==============================")
        print("Growth Stage:", growth_stage)
        print("Total Tomatoes:", total_tomatoes)
        print("Ripe:", ripe_count)
        print("Half-ripe:", halfripe_count)
        print("Unripe:", unripe_count)

                # ----------------------------------
        # SAVE RESULTS TO CSV
        # ----------------------------------

        timestamp = datetime.now().strftime("%H:%M:%S")

        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                timestamp,
                growth_stage,
                total_tomatoes,
                ripe_count,
                halfripe_count,
                unripe_count
            ])


          # ======================================
    # DISPLAY CURRENT FRAME
    # ======================================

    # Keep the latest YOLO detection frame visible
    if detected_frame is not None:
        display_frame = detected_frame.copy()
    else:
        display_frame = frame.copy()

    frame_small = cv2.resize(
        display_frame,
        (960, 540)
    )


    # ======================================
    # PUT AI RESULTS ON VIDEO
    # ======================================

    cv2.putText(
        frame_small,
        "GreenVision AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_small,
        f"Growth Stage: {growth_stage}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_small,
        f"Total Tomatoes: {total_tomatoes}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_small,
        f"Ripe: {ripe_count}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_small,
        f"Half-ripe: {halfripe_count}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_small,
        f"Unripe: {unripe_count}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # SHOW VIDEO
    # ======================================

    cv2.imshow("GreenVision AI", frame_small)

    # Press Q to stop
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break


# ==========================================
# CLEAN UP
# ==========================================

video.release()
cv2.destroyAllWindows()

print()
print("GreenVision video analysis finished.")
print("Results saved to:", csv_file)