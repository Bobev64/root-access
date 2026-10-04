from ultralytics import YOLO

print("Starting GreenVision...")

# ==========================================
# 1. LOAD OUR TWO AI MODELS
# ==========================================

# Detects tomatoes and their ripeness
ripeness_model = YOLO("models/tomato_ripeness.pt")

# Classifies the plant's growth stage
growth_model = YOLO("models/tomato_growth.pt")


# ==========================================
# 2. IMAGE TO ANALYZE
# ==========================================

image_path = "tomatoleaf.jpg"


# ==========================================
# 3. TOMATO + RIPENESS DETECTION
# ==========================================

ripeness_results = ripeness_model(image_path, conf=0.5)
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


# Calculate total tomatoes
total_tomatoes = (
    ripe_count
    + halfripe_count
    + unripe_count
)


# ==========================================
# 4. DETERMINE GROWTH STAGE
# ==========================================

# If tomatoes are visible, we already know
# the plant has reached the fruiting stage.
if total_tomatoes > 0:

    growth_stage = "Fruiting / Ripening"

else:

    # No tomatoes detected, so ask our
    # growth-stage classification model.
    growth_results = growth_model(image_path)
    growth_result = growth_results[0]

    # Get the class with the highest probability
    class_id = growth_result.probs.top1

    growth_stage = growth_model.names[class_id]


# Make class names prettier
if growth_stage == "Early_Vegetative":
    growth_stage = "Early Vegetative"

elif growth_stage == "Flowering_Initiation":
    growth_stage = "Flowering Initiation"


# ==========================================
# 5. PRINT GREENVISION RESULTS
# ==========================================

print()
print("================================")
print("       GREENVISION RESULTS")
print("================================")
print()

print("Growth Stage:", growth_stage)

print()
print("Total Tomatoes:", total_tomatoes)

print()
print("Ripe:", ripe_count)
print("Half-ripe:", halfripe_count)
print("Unripe:", unripe_count)

print()
print("================================")


# ==========================================
# 6. SHOW IMAGE WITH DETECTION BOXES
# ==========================================

ripeness_result.show()