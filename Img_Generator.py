import cv2
import os
import numpy as np
import random
import math

# Ensure the output directory exists
output_dir = "Dataset_1"
os.makedirs(output_dir, exist_ok=True)

#=================
# Squares
#=================
for i in range(500):  # Generate 500 images
    # 1. Create a white square canvas
    file_path = f"Dataset_1/img{i}.png"

    width, height = 128, 128
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    num_squares = 1
    min_size = 10  # Minimum side length in pixels
    max_size = 70  # Maximum side length in pixels
    safety_margin = 8 

    for _ in range(num_squares):
        while True:
            size = random.randint(min_size, max_size)
            # Step A: Pick a completely random first vertex
            x1 = random.randint(safety_margin, width - size - safety_margin)
            y1 = random.randint(safety_margin, height - size - safety_margin)
        
            ## Step C: Calculate the remaining 3 corners based on the top-left corner and size
            x2, y2 = x1 + size, y1          # Top-right
            x3, y3 = x1 + size, y1 + size   # Bottom-right
            x4, y4 = x1, y1 + size          # Bottom-left
            
            # Double check all points fall within our safety zone (optional, but keeps logic consistent)
            if (safety_margin <= x1 and x3 <= width - safety_margin and
                safety_margin <= y1 and y3 <= height - safety_margin):
                break # Valid square found!

        #Bundle the 4 points together and reshape for OpenCV
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32).reshape((-1, 1, 2))
    
        
        # Random BGR Color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
        # Draw the square
        cv2.fillPoly(canvas, [pts], color)

        # 2. Define outline properties
        outline_color = (0, 0, 0) # Black outline
        thickness = 2             # Thickness in pixels
        is_closed = True          # Connects the last point back to the first point
    
        # 3. Draw the outline
        cv2.polylines(canvas, [pts], is_closed, outline_color, thickness)

        cv2.imwrite(file_path, canvas)

#=================
# Circles
#=================
for i in range(500):  # Generate 500 images
    file_path = f"{output_dir}/img{i + 500}.png"

    # 1. Create a white square canvas
    width, height = 128, 128
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    num_circles = 1
    min_radius = 5  # Minimum radius (20px total width)
    max_radius = 35  # Maximum radius (70px total width)
    safety_margin = 8 

    for _ in range(num_circles):
        while True:
            # Step A: Pick a random radius size
            radius = random.randint(min_radius, max_radius)
            
            # Step B: Pick a random center point (cx, cy)
            # We restrict the range based on the dynamic radius size so it never spills over
            cx = random.randint(safety_margin + radius, width - radius - safety_margin)
            cy = random.randint(safety_margin + radius, height - radius - safety_margin)
            
            # Double check boundaries to be 100% safe
            if (safety_margin <= cx - radius and cx + radius <= width - safety_margin and
                safety_margin <= cy - radius and cy + radius <= height - safety_margin):
                break # Valid circle placement found!

        # Random BGR Color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
        # Draw the filled circle
        # -1 thickness fills the circle completely
        cv2.circle(canvas, (cx, cy), radius, color, thickness=-1)

        # Define outline properties
        outline_color = (0, 0, 0) # Black outline
        outline_thickness = 2     # Thickness in pixels
    
        # Draw the outline circle
        cv2.circle(canvas, (cx, cy), radius, outline_color, thickness=outline_thickness)

    cv2.imwrite(file_path, canvas)

#=================
# Triangles
#=================
for i in range(500):  # Generate 500 images
    # 1. Create a white square canvas
    file_path = f"Dataset_1/img{i+1000}.png"

    width, height = 128, 128
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    num_triangles = 1
    min_size = 10  # Minimum side length in pixels
    max_size = 70  # Maximum side length in pixels

    for _ in range(num_triangles):
        while True:
            # Step A: Pick a completely random first vertex
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
        
            # Vertex 2
            angle2 = random.uniform(0, 2 * math.pi)
            distance2 = random.randint(min_size, max_size)
            x2 = int(x1 + distance2 * math.cos(angle2))
            y2 = int(y1 + distance2 * math.sin(angle2))
        
            # Vertex 3
            angle3 = angle2 + random.uniform(math.pi / 4, 7 * math.pi / 4) 
            distance3 = random.randint(min_size, max_size)
            x3 = int(x1 + distance3 * math.cos(angle3))
            y3 = int(y1 + distance3 * math.sin(angle3))
            
            # Step B: Enforce a strict buffer zone from the 128x128 edges
            # A safety_margin of 8 keeps the triangle and its thick border perfectly isolated
            safety_margin = 8 
            
            if (safety_margin <= x1 <= width - safety_margin and safety_margin <= y1 <= height - safety_margin and
                safety_margin <= x2 <= width - safety_margin and safety_margin <= y2 <= height - safety_margin and
                safety_margin <= x3 <= width - safety_margin and safety_margin <= y3 <= height - safety_margin):
                break # Valid triangle found!

        # Bundle points and reshape for OpenCV
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32).reshape((-1, 1, 2))
    
        # Random BGR Color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
        # Draw the triangle
        cv2.fillPoly(canvas, [pts], color)

        # 2. Define outline properties
        outline_color = (0, 0, 0) # Black outline
        thickness = 2             # Thickness in pixels
        is_closed = True          # Connects the last point back to the first point
    
        # 3. Draw the outline
        cv2.polylines(canvas, [pts], is_closed, outline_color, thickness)

    cv2.imwrite(file_path, canvas)
