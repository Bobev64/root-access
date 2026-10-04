import sys
from ultralytics import YOLO


# ==========================================
# LOAD GREENVISION AI MODELS
# ==========================================

ripeness_model = YOLO("models/tomato_ripeness.pt")
growth_model = YOLO("models/tomato_growth.pt")


# ==========================================
# ANALYZE A TOMATO PLANT
# ==========================================

def analyze_plant(image_path):

    # --------------------------------------
    # TOMATO + RIPENESS DETECTION
    # --------------------------------------

    ripeness_results = ripeness_model(
        image_path,
        conf=0.5,
        verbose=False
    )

    ripeness_result = ripeness_results[0]

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


    # --------------------------------------
    # DETERMINE GROWTH STAGE
    # --------------------------------------

    if total_tomatoes > 0:

        growth_stage = "Fruiting / Ripening"

    else:

        growth_results = growth_model(
            image_path,
            verbose=False
        )

        growth_result = growth_results[0]

        class_id = growth_result.probs.top1
        growth_stage = growth_model.names[class_id]


        if growth_stage == "Early_Vegetative":
            growth_stage = "Early Vegetative"

        elif growth_stage == "Flowering_Initiation":
            growth_stage = "Flowering Initiation"


    # --------------------------------------
    # STRUCTURED RESULTS
    # --------------------------------------

    results = {
        "growth_stage": growth_stage,
        "total_tomatoes": total_tomatoes,
        "ripe": ripe_count,
        "half_ripe": halfripe_count,
        "unripe": unripe_count
    }

    return results, ripeness_result


# ==========================================
# RUN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("Starting GreenVision...")

    if len(sys.argv) < 2:
        print("Please provide an image.")
        print(
            "Example: python main.py tomatoplantreal.jpg"
        )
        sys.exit()

    image_path = sys.argv[1]

    results, detection_image = analyze_plant(image_path)


    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    print()
    print("================================")
    print("       GREENVISION RESULTS")
    print("================================")
    print()

    print("Growth Stage:", results["growth_stage"])

    print()
    print("Total Tomatoes:", results["total_tomatoes"])

    print()
    print("Ripe:", results["ripe"])
    print("Half-ripe:", results["half_ripe"])
    print("Unripe:", results["unripe"])

    print()
    print("================================")


    # Show image with YOLO detection boxes
    detection_image.show()