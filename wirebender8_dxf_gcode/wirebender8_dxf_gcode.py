import ezdxf
import math
from scipy.interpolate import splprep, splev
import numpy as np
import re
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
input_dxf = "RV.dxf"
output_file = "Gcode\\wire_bender_output.gcode"
output_final = "Gcode\\wire_marlin.gcode"

#IGNORE (DOESN'T AFFECT CODE CONVERSION FOR KLIPPER)
feed_rate = 100          # mm/min feed
rotation_speed = 30      # deg/sec bending

#DOES AFFECT CODE CONVERSION FOR KLIPPER
curve_resolution_mm = 9  #how large each segment is extruded before a bend, smaller=smoother
spring_back = -2         #degree (offset for springback)
feedrate_bend = 1600     #feedrate for the X-axis motor (1600/60)mm/s
feedrate_pin = 800       #feedrate for the bending pin (800/60)mm/s
feedrate_wire = 1800     #feedrate for the wire extruding (1800/60)mm/s

# ---------- GEOMETRY ----------
def distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def vector_angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    det = v1[0]*v2[1] - v1[1]*v2[0]
    return math.degrees(math.atan2(det, dot))

def interpolate_points(points, resolution_mm):
    if len(points) < 2:
        return points
    resampled = [points[0]]
    accumulated = 0.0
    for i in range(1, len(points)):
        prev, curr = points[i-1], points[i]
        seg_len = distance(prev, curr)
        accumulated += seg_len
        if accumulated >= resolution_mm:
            resampled.append(curr)
            accumulated = 0.0
    if resampled[-1] != points[-1]:
        resampled.append(points[-1])
    return resampled

# ---------- DXF SEGMENT HANDLERS ----------
def segment_arc_from_dxf(arc_entity, resolution_mm=1.0):
    cx, cy = arc_entity.dxf.center.x, arc_entity.dxf.center.y   
    r = arc_entity.dxf.radius
    start_angle_deg = arc_entity.dxf.start_angle
    end_angle_deg = arc_entity.dxf.end_angle

    start_pt = (cx + r * math.cos(math.radians(start_angle_deg)),
                cy + r * math.sin(math.radians(start_angle_deg)))
    end_pt = (cx + r * math.cos(math.radians(end_angle_deg)),
                cy + r * math.sin(math.radians(end_angle_deg)))

    
    sx, sy = start_pt[0] - cx, start_pt[1] - cy
    ex, ey = end_pt[0] - cx, end_pt[1] - cy

    cross = sx * ey - sy * ex

    if cross < 0:
        s = math.radians(start_angle_deg)
        e = math.radians(end_angle_deg)
        delta = (e - s) % (2 * math.pi)
        arc_length = r * delta
        num_points = max(3, int((arc_length) / resolution_mm))
        
        points = []
    else:
        e = math.radians(start_angle_deg)
        s = math.radians(end_angle_deg)
        delta = (e - s) % (2 * math.pi)
        delta = delta - (2 * math.pi )
        arc_length = r * delta
        num_points = max(3, int((-1*arc_length) / resolution_mm))
        
        points = []

    for i in range(num_points + 1):
        a = s + delta * i / num_points
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        points.append((x, y))
        

    return points

def segment_spline(spline, resolution_mm):
    if spline.fit_points:
        pts = [(p[0], p[1]) for p in spline.fit_points]
    elif spline.control_points:
        pts = [(p[0], p[1]) for p in spline.control_points]
    else:
        pts = [(spline.start_point.x, spline.start_point.y),
                (spline.end_point.x, spline.end_point.y)]

    if len(pts) < 2:
        return pts

    x, y = np.array(pts).T
    tck, _ = splprep([x, y], s=0, k=min(3, len(pts)-1))
    total_len = sum(distance(pts[i], pts[i+1]) for i in range(len(pts)-1))
    num_points = max(2, int(total_len / resolution_mm))
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return list(zip(x_new, y_new))

# ---------- DXF PARSING ----------
def extract_segments_from_dxf(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    entities = sorted(msp, key=lambda e: int(e.dxf.handle,16))
    segments = []

    for e in entities:
        pts = []
        etype = e.dxftype()
        if etype == "LINE":
            pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        elif etype in ("LWPOLYLINE", "POLYLINE"):
            pts = [(p[0], p[1]) for p in e.get_points()]
        elif etype == "ARC":
            pts = segment_arc_from_dxf(e, resolution_mm=curve_resolution_mm)
        elif etype == "SPLINE":
            pts = segment_spline(e, curve_resolution_mm)

        if pts:
            pts = interpolate_points(pts, curve_resolution_mm)
            segments.append(pts)

    return segments

# ---------- MAIN PROCESS ----------
def generate_wire_bender_from_dxf(input_dxf, output_file):
    segments = extract_segments_from_dxf(input_dxf)
    if not segments:
        raise ValueError("DXF contains no valid segments.")

    with open(output_file, "w") as f:
        f.write("; G-code generated from DXF for 3D wire bender\n")

        prev_point = None
        prev_prev_point = None

        for segment in segments:
            if not segment:
                continue

            f.write("; Start new segment\n")

            # ---- INITIAL ORIENTATION FIX ----
            if prev_point is None and len(segment) >= 2:
                x0, y0 = segment[0]
                x1, y1 = segment[1]

                v = (x1 - x0, y1 - y0)

                # Angle of first segment relative to horizontal
                initial_angle = vector_angle((1, 0), v)

                if abs(initial_angle) > 0.1:
                    f.write(f"; Initial alignment to match DXF first segment\n")
                    f.write(f"A{initial_angle:.2f} F{rotation_speed}\n")

            # Inter-segment bend
            if prev_point and prev_prev_point:
                v_prev = (prev_point[0]-prev_prev_point[0], prev_point[1]-prev_prev_point[1])
                v_next = (segment[1][0]-segment[0][0], segment[1][1]-segment[0][1])
                if distance((0,0), v_prev) > 1e-6 and distance((0,0), v_next) > 1e-6:
                    ang = vector_angle(v_prev, v_next)
                    if abs(ang) > 0.1:
                        f.write(f"; Bend {ang:.2f} degrees (segment transition)\n")
                        f.write(f"A{ang:.2f} F{rotation_speed}\n")

            # Feed moves
            for i in range(1, len(segment)):
                x_prev, y_prev = segment[i-1]
                x_curr, y_curr = segment[i]
                feed_dist = distance((x_prev, y_prev), (x_curr, y_curr))
                if feed_dist < 1e-6:
                    continue
                f.write(f"G1 X{feed_dist:.2f} Z0.00 F{feed_rate}\n")

                # Internal bends
                if i < len(segment)-1:
                    x_next, y_next = segment[i+1]
                    v1 = (x_curr - x_prev, y_curr - y_prev)
                    v2 = (x_next - x_curr, y_next - y_curr)
                    if distance((0,0), v1) < 1e-6 or distance((0,0), v2) < 1e-6:
                        continue
                    ang = vector_angle(v1, v2)
                    if abs(ang) > 0.1:
                        f.write(f"; Bend {ang:.2f} degrees\n")
                        f.write(f"A{ang:.2f} F{rotation_speed}\n")

            prev_prev_point = segment[-2] if len(segment) >= 2 else prev_point
            prev_point = segment[-1]


generate_wire_bender_from_dxf(input_dxf, output_file)






# ============================================
# Parse G-code
# ============================================

def parse_gcode(filename):
   
    commands = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines or comments
            if not line or line.startswith(';'):
                continue

            # ---- Parse G1 moves ----
            if line.startswith("G1"):
                # extract X and Z values if they exist
                x_match = re.search(r'X([-0-9.]+)', line)
                z_match = re.search(r'Z([-0-9.]+)', line)

                x_val = float(x_match.group(1)) if x_match else 0.0
                z_val = float(z_match.group(1)) if z_match else 0.0

                # distance is feed length = magnitude of X/Z vector
                distance = math.sqrt(x_val**2 + z_val**2)

                commands.append(("G1", distance))

            # ---- Parse A bends ----
            elif line.startswith("A"):
                # Extract angle like A-93.24
                a_match = re.match(r'A([-0-9.]+)', line)
                if a_match:
                    angle = float(a_match.group(1))
                    commands.append(("A", angle))

    return commands


# ============================================
# Simulate wire bending path
# ============================================

def simulate_path(commands, start_angle=0):
    x, z = 0.0, 0.0
    angle_deg = start_angle

    path_x = [x]
    path_z = [z]

    for cmd, val in commands:
        if cmd == "G1":
            # Feed wire forward
            x += val * math.cos(math.radians(angle_deg))
            z += val * math.sin(math.radians(angle_deg))
            path_x.append(x)
            path_z.append(z)

        elif cmd == "A":
            # Rotate wire bend direction
            angle_deg += val

    return path_x, path_z


# ============================================
# MAIN
# ============================================


commands = parse_gcode(output_file)

px, pz = simulate_path(commands)

# ============================================
# Plot result
# ============================================  

plt.figure(figsize=(8,6))
plt.plot(px, pz, '-o', linewidth=2, markersize=5)
plt.title("Wire Bender Path Visualization (From G-code)")
plt.xlabel("X (mm)")
plt.ylabel("Z (mm)")
plt.grid(True)
plt.axis('equal')
plt.show()  












# ============================================
# Convert DXF-generated wire bender G-code to KLIPPER G-code
# ============================================


# ---------- BEND FORMULAS FOR KLIPPER ----------
def marlin_cw_formula(angle_deg):
    """CW bends (angle < 0)"""
    x = angle_deg + spring_back
    return (0.000000009495*x**5 - 0.000003234*x**4 + 0.0003487*x**3 - 0.01445*x**2 + 0.09436*x + 47.27)

def marlin_ccw_formula(angle_deg):
    """CCW bends (angle > 0)"""
    x = (-1*angle_deg) + spring_back
    return ( -0.000000008079*x**5 + 0.00000291*x**4
            - 0.0003346*x**3 + 0.01534*x**2
            - 0.1524*x + 42.63)

# ---------- KLIPPER CONVERSION FUNCTION ----------
def convert_to_marlin_gcode(input_file_prev, output_final=None,
                             feed_axis="E", bend="X", pin_axis="Z", feed_bend=feedrate_bend, feed_pin=feedrate_pin):
   
    if output_final is None:
        base, ext = os.path.splitext(input_file_prev)
        output_final = base + "_marlin" + ext

    marlin_bend_pos = 0.0
    last_bend_type = None

    # -------- Read all input lines --------
    with open(input_file_prev, "r") as f_in:
        lines = [line.rstrip() for line in f_in]

     # -------- Remove first bend if it is the very first real command --------
    lines_modified = lines.copy()
    for i, line in enumerate(lines_modified):
        stripped = line.strip()
    
        # Skip blank lines and comments
        if not stripped or stripped.startswith(";"):
            continue
    
       
        if stripped.startswith("A"):
            print(f"Removing first bend command: {stripped}")
            lines_modified.pop(i)
        
        break


    with open(output_final, "w") as f_out:
        # HOMING for KLIPPER
        f_out.write("; Converted for KLIPPER 3D printer\n")
        f_out.write("G28 Z ; home axes\n")
        f_out.write("G28 X Y ; home axes\n")
        f_out.write("G90\n")
        f_out.write("M83\n")
        f_out.write("M106 S255\n")
        f_out.write("G1 X35 Y78 F1600\n")
        f_out.write("G4 P500\n")
        

      

        for line in lines_modified:
            if not line:
                continue

            # ---- Feed moves ----
            if line.startswith("G1"):
                x_match = re.search(r"X([-0-9.]+)", line)
                x_val = float(x_match.group(1)) if x_match else 0.0
                f_out.write(f"G1 {feed_axis}{x_val:.4f} F{feedrate_wire}\n")

            # ---- Bend moves ----
            elif line.startswith("A"):
                a_match = re.match(r"A([-0-9.]+)", line)
                if not a_match:
                    continue
                angle = float(a_match.group(1))
                if angle > 0:
                    bend_type = "CW"
                    marlin_bend_pos = marlin_cw_formula(angle)
                else:
                    bend_type = "CCW"
                    marlin_bend_pos = marlin_ccw_formula(angle)

                

                # Only do pin/X sequence if the bend type changed
                if bend_type != last_bend_type:
                    if bend_type == "CW":
                        f_out.write(f"G1 {pin_axis}0 F{feed_pin}; lower pin\n")
                        f_out.write(f"G1 {bend}55 F2000\n")
                        f_out.write(f"G1 {pin_axis}8 F{feed_pin}; raise pin\n")
                        f_out.write(f"G1 {bend}49 F{feed_bend}\n")
                        f_out.write(f"G1 {bend}{marlin_bend_pos:.4f} F{feed_bend} ; original angle {angle:.2f}\n")
                        f_out.write(f"G1 {bend}55 F{feed_bend}\n")
                        
                        
                    elif bend_type == "CCW":  # CCW1
                        f_out.write(f"G1 {pin_axis}0 F{feed_pin}; lower pin\n")
                        f_out.write(f"G1 {bend}35 F2000\n")
                        f_out.write(f"G1 {pin_axis}8 F{feed_pin}; raise pin\n")
                        f_out.write(f"G1 {bend}40 F{feed_bend}\n")
                        f_out.write(f"G1 {bend}{marlin_bend_pos:.4f} F{feed_bend} ; original angle {angle:.2f}\n")
                        f_out.write(f"G1 {bend}35 F{feed_bend}\n")
                        
                else:
                    if bend_type == "CW":
                        f_out.write(f"G1 {bend}49 F{feed_bend}\n") 
                        f_out.write(f"G1 {bend}{marlin_bend_pos:.4f} F{feed_bend} ; original angle {angle:.2f}\n")
                        f_out.write(f"G1 {bend}55 F{feed_bend}\n")

                    elif bend_type == "CCW":  # CCW2
                        f_out.write(f"G1 {bend}40 F{feed_bend}\n") 
                        f_out.write(f"G1 {bend}{marlin_bend_pos:.4f} F{feed_bend} ; original angle {angle:.2f}\n")
                        f_out.write(f"G1 {bend}35 F{feed_bend}\n")
                    

                last_bend_type = bend_type

        f_out.write(f"G1 {pin_axis}0 F{feed_pin}; lower pin\n")
        f_out.write("G1 X35 Y78 F1600\n")
        f_out.write(f"SET_STEPPER_ENABLE STEPPER=extruder ENABLE=0\n")
        f_out.write("M18\n")
        f_out.write("M84\n")

    print(f"KLIPPER-compatible G-code saved as:\n{output_final}")

    
convert_to_marlin_gcode(output_file, output_final)




