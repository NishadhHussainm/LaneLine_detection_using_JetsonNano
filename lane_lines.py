import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    if lines is None:
        return
    for line in lines:
        if line is not None:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def average_slope_intercept(lines):
    left_lines = []
    right_lines = []
    left_weights = []
    right_weights = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            
            if slope < -0.5:  # Left lane: slope should be negative and reasonably steep
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > 0.5:  # Right lane: slope should be positive and reasonably steep
                right_lines.append((slope, intercept))
                right_weights.append(length)
    
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    if slope == 0:  # Avoid division by zero
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def lane_detection(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100)
    
    if lines is not None:
        left_line, right_line = average_slope_intercept(lines)
        
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        
        left_line_points = make_line_points(y1, y2, left_line)
        right_line_points = make_line_points(y1, y2, right_line)
        
        draw_lines(image, [left_line_points, right_line_points])
    
    return image

cap = cv2.VideoCapture('/dev/video0')  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    lane_image = lane_detection(frame)
    
    cv2.imshow('Lane Detection', lane_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

