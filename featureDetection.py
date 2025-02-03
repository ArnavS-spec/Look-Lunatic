import cv2
import mediapipe as mp
import math
import numpy as np
from scipy.spatial import ConvexHull

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LANDMARK_INDICES = {
    'jawline': [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 
               378, 379, 365, 397, 288, 435, 361, 454, 323, 365],
    'eyes': {
        'left': [33, 160, 158, 133, 153, 144, 145, 7, 163, 246],
        'right': [362, 385, 387, 263, 373, 380, 374, 8, 390, 466]
    },
    'eyebrows': {
        'left': [70, 63, 105, 66, 107, 55, 65, 53],
        'right': [336, 296, 334, 293, 300, 285, 295, 283]
    },
    'lips': {
        'outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
        'inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
    },
    'reference': {
        'face_width': [234, 454],
        'nose_tip': [4],
        'forehead': [10]
    }
}

def get_normalized_coords(face_landmarks, indices, frame_shape):
    h, w = frame_shape[:2]
    return [(face_landmarks.landmark[i].x * w, 
             face_landmarks.landmark[i].y * h) 
            for i in indices]

def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    angle = np.degrees(np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))))
    return angle

def analyze_eyebrow_shape(points):
    # Multi-stage shape analysis arnav
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    
    # 1. Polynomial curvature analysis
    coeffs = np.polyfit(x, y, 3)
    curvature = 2 * coeffs[0] / (1 + (coeffs[1] + 3*coeffs[0]*x)**2)**1.5
    avg_curvature = np.mean(np.abs(curvature))
    
    # 2. Slope dynamics
    slopes = np.diff(y)/np.diff(x)
    slope_var = np.var(slopes)
    
    # 3. Vertical displacement
    vertical_diff = points[-1][1] - points[0][1]
    
    # 4. Convex hull analysis
    hull = ConvexHull(points)
    hull_ratio = hull.area / (hull.volume + 1e-6)
    
    # Classification logic
    if avg_curvature > 0.025:
        if np.any(curvature < -0.01) and np.any(curvature > 0.01):
            return "S-shaped"
        return "Curved"
    elif slope_var > 0.015:
        if vertical_diff < -2.3: return "Ascending"
        if vertical_diff > 3: return "Descending"
        return "Angled"
    elif hull_ratio > 0.4:
        return "Rounded"
    elif np.abs(slopes).mean() < 0.05:
        return "Straight"
    return "Soft Arched"

def analyze_eye_shape(eye_points, face_width):
    eye_width = math.dist(eye_points[0], eye_points[3])
    eye_height = np.mean([math.dist(p, eye_points[4]) for p in eye_points[4:6]])
    aspect_ratio = eye_height / eye_width
    
    crease_depth = abs(eye_points[6][1] - eye_points[1][1])
    
    inner_angle = calculate_angle(eye_points[0], eye_points[1], eye_points[2])
    outer_angle = calculate_angle(eye_points[3], eye_points[4], eye_points[5])
    
    if crease_depth < 2:
        return "Hooded"
    elif aspect_ratio > 0.38:
        return "Round"
    elif outer_angle - inner_angle > 20:
        return "Upturned"
    elif inner_angle - outer_angle > 15:
        return "Downturned"
    elif face_width/eye_width > 4.5:
        return "Wide-set"
    return "Almond"

def analyze_jawline_shape(jaw_points, face_width):
    # Enhanced 3D contour analysis
    chin_idx = len(jaw_points) // 2
    angles = []
    curvature_scores = []
    
    for i in range(1, chin_idx):
        angle = calculate_angle(jaw_points[i-1], jaw_points[i], jaw_points[i+1])
        angles.append(angle)
        curvature = 1/(math.dist(jaw_points[i-1], jaw_points[i+1]) + 1e-6)
        curvature_scores.append(curvature)
    
    # Dynamic threshold calculation
    angle_threshold = np.mean(angles) * 0.85
    sharp_angles = sum(1 for a in angles if a < angle_threshold)
    
    # Width ratios
    jaw_width = math.dist(jaw_points[0], jaw_points[-1])
    chin_width = math.dist(jaw_points[chin_idx-3], jaw_points[chin_idx+3])
    
    # Classification matrix
    if sharp_angles > 4 and chin_width/jaw_width < 0.65:
        return "Square"
    elif np.mean(curvature_scores) > 0.12:
        return "Angular"
    elif chin_width/jaw_width > 0.82:
        return "Round"
    elif face_width/jaw_width > 1.35:
        return "Heart"
    elif chin_width/jaw_width < 0.6:
        return "Diamond"
    return "Oval"

def analyze_lip_shape(outer_points, inner_points, face_width):
    # Precision geometric analysis
    # 1. Volume ratios
    upper_hull = ConvexHull(outer_points[:7])
    lower_hull = ConvexHull(outer_points[6:])
    volume_ratio = upper_hull.volume / (lower_hull.volume + 1e-6)
    
    # 2. Angular dynamics
    left_angle = calculate_angle(outer_points[0], outer_points[1], outer_points[3])
    right_angle = calculate_angle(outer_points[-1], outer_points[-2], outer_points[-4])
    
    # 3. Cupid's bow analysis
    bow_height = outer_points[3][1] - min(outer_points[2][1], outer_points[4][1])
    bow_width = math.dist(outer_points[2], outer_points[4])
    
    # 4. Corner dynamics
    left_corner_slope = (outer_points[1][1] - outer_points[0][1]) / (outer_points[1][0] - outer_points[0][0])
    right_corner_slope = (outer_points[-2][1] - outer_points[-1][1]) / (outer_points[-1][0] - outer_points[-2][0])
    
    # Classification logic
    if (left_angle + right_angle) > 185:
        return "Downturned"
    elif bow_height > 4 and bow_width < 18:
        return "Bow-shaped"
    elif volume_ratio > 1.7:
        return "Heavy Upper"
    elif volume_ratio < 0.6:
        return "Heavy Lower"
    elif abs(left_corner_slope - right_corner_slope) > 0.3:
        return "Asymmetric"
    elif (outer_points[-1][0] - outer_points[0][0])/face_width > 0.47:
        return "Wide"
    elif lower_hull.volume > upper_hull.volume*1.4:
        return "Full Lower"
    return "Balanced Full"

def analyze_face(face_landmarks, frame_shape):
    coords = {}
    for category, indices in LANDMARK_INDICES.items():
        if isinstance(indices, dict):
            coords[category] = {k: get_normalized_coords(face_landmarks, v, frame_shape) 
                               for k, v in indices.items()}
        else:
            coords[category] = get_normalized_coords(face_landmarks, indices, frame_shape)
    
    face_width = math.dist(*coords['reference']['face_width'])
    
    return {
        'eyebrows': {
            'left': analyze_eyebrow_shape(coords['eyebrows']['left']),
            'right': analyze_eyebrow_shape(coords['eyebrows']['right'])
        },
        'eyes': {
            'left': analyze_eye_shape(coords['eyes']['left'], face_width),
            'right': analyze_eye_shape(coords['eyes']['right'], face_width)
        },
        'jawline': analyze_jawline_shape(coords['jawline'], face_width),
        'lips': analyze_lip_shape(coords['lips']['outer'], coords['lips']['inner'], face_width)
    }

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        
        frame = cv2.flip(frame, 1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            analysis = analyze_face(results.multi_face_landmarks[0], frame.shape)
            
            y_offset = 40
            categories = [
                ('Jawline', analysis['jawline']),
                ('Left Eye', analysis['eyes']['left']),
                ('Right Eye', analysis['eyes']['right']),
                ('Left Eyebrow', analysis['eyebrows']['left']),
                ('Right Eyebrow', analysis['eyebrows']['right']),
                ('Lip Shape', analysis['lips'])
            ]
            
            for cat_name, cat_value in categories:
                text = f"{cat_name}: {cat_value}"
                cv2.putText(frame, text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
                y_offset += 35
        
        cv2.imshow('Facial Feature Analysis', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
