import dearpygui.dearpygui as dpg
import numpy as np

# --- Data storage ---
points = []  # [[x, y], ...] in grid coordinates
bars = []  # [[point_idx1, point_idx2], ...]
explicit_joints = []  # [[bar_idx1, bar_idx2], ...]
fixed = []  # [[bar_idx, distance_along_bar], ...]

mode = "points"
selected_points = []
selected_bars = []

CANVAS_SIZE = 800
GRID_SPACING = 50  # pixels
grid_scale = 1.0  # scaling factor for export

ref_width = 0
ref_height = 0
# --- Helper functions ---
def canvas_to_grid(pos):
    """Convert canvas pixel pos to grid coordinates with center at (0,0)."""
    cx = pos[0] - CANVAS_SIZE/2
    cy = pos[1] - CANVAS_SIZE/2
    return np.array([cx, -cy]) / GRID_SPACING  # invert y for conventional coordinates

def grid_to_canvas(pos):
    """Convert grid coordinates to canvas pixels."""
    x = pos[0]*GRID_SPACING + CANVAS_SIZE/2
    y = -pos[1]*GRID_SPACING + CANVAS_SIZE/2
    return np.array([x, y])


def get_mouse_pos_on_canvas():
    mouse = np.array(dpg.get_mouse_pos())  # screen coordinates
    rect_min = np.array(dpg.get_item_rect_min("canvas_drawlist"))  # top-left corner in screen coords
    rect_max = np.array(dpg.get_item_rect_max("canvas_drawlist"))  # bottom-right

    # Relative mouse position inside drawlist
    mouse_rel = mouse - np.array([7,7])
    #mouse_rel[mouse_rel < 0] = 0

    width = rect_max[0] - rect_min[0]
    height = rect_max[1] - rect_min[1]

    # Clamp to drawlist
    if 0 <= mouse_rel[0] <= width and 0 <= mouse_rel[1] <= height:
        return mouse_rel
    return None

def bar_intersection(bar_idx1, bar_idx2):
    b1 = bars[bar_idx1]
    b2 = bars[bar_idx2]
    p1, p2 = np.array(points[b1[0]]), np.array(points[b1[1]])
    p3, p4 = np.array(points[b2[0]]), np.array(points[b2[1]])
    # shared point
    shared = None
    for pt1 in [p1, p2]:
        for pt2 in [p3, p4]:
            if np.allclose(pt1, pt2):
                shared = pt1
    if shared is not None:
        return shared
    denom = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
    if abs(denom) < 1e-8:
        return (p1+p2)/2
    x = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[0]-p4[0]) - (p1[0]-p2[0])*(p3[0]*p4[1]-p3[1]*p4[0])) / denom
    y = ((p1[0]*p2[1]-p1[1]*p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]*p4[1]-p3[1]*p4[0])) / denom
    return np.array([x, y])

def get_implicit_joints():
    joints = []
    for i, b1 in enumerate(bars):
        for j, b2 in enumerate(bars):
            if i >= j: continue
            if len(set(b1) & set(b2)) > 0:
                joints.append([i, j])
    return joints

def draw_grid():
    # vertical
    for x in range(0, CANVAS_SIZE, GRID_SPACING):
        dpg.draw_line([x, 0], [x, CANVAS_SIZE], color=(200,200,200,255), thickness=1, parent="canvas_drawlist")
    # horizontal
    for y in range(0, CANVAS_SIZE, GRID_SPACING):
        dpg.draw_line([0, y], [CANVAS_SIZE, y], color=(200,200,200,255), thickness=1, parent="canvas_drawlist")
    # center lines
    cx = CANVAS_SIZE/2
    cy = CANVAS_SIZE/2
    dpg.draw_line([cx, 0], [cx, CANVAS_SIZE], color=(150,0,0,255), thickness=1, parent="canvas_drawlist")
    dpg.draw_line([0, cy], [CANVAS_SIZE, cy], color=(150,0,0,255), thickness=1, parent="canvas_drawlist")

# --- Reference image storage ---
ref_texture = None
ref_image = None  # actual canvas draw info
ref_filename = ""
ref_offset = np.array([0.0, 0.0])
ref_scale = 1.0
ref_alpha = 128  # 0-255

# --- Reference image functions ---
def draw_canvas():
    dpg.delete_item("canvas_drawlist", children_only=True)
    draw_grid()

    # Draw reference image first
    if ref_texture is not None:
        w = ref_width * ref_scale
        h = ref_height * ref_scale

        canvas_coords_min = [
            CANVAS_SIZE / 2 + ref_offset[0],
            CANVAS_SIZE / 2 + ref_offset[1]
        ]

        canvas_coords_max = [
            canvas_coords_min[0] + w,
            canvas_coords_min[1] + h
        ]

        dpg.draw_image(
            ref_texture,
            canvas_coords_min,
            canvas_coords_max,
            parent="canvas_drawlist",
            color=(255, 255, 255, ref_alpha)
        )

    # --- Bars ---
    for i, b in enumerate(bars):
        p1 = grid_to_canvas(points[b[0]])
        p2 = grid_to_canvas(points[b[1]])
        color = (0,0,255,255)
        if i in selected_bars: color=(0,255,0,255)
        dpg.draw_line(p1, p2, parent="canvas_drawlist", color=color, thickness=2)
    # --- Points ---
    for i, p in enumerate(points):
        p_canvas = grid_to_canvas(p)
        color = (255,0,0,255)
        if i in selected_points: color=(0,255,0,255)
        dpg.draw_circle(p_canvas, 5, color=color, fill=color, parent="canvas_drawlist")
    # --- Joints ---
    all_joints = explicit_joints + get_implicit_joints()
    for j in all_joints:
        b1 = bars[j[0]]
        b2 = bars[j[1]]
        pos = grid_to_canvas(bar_intersection(j[0], j[1]))
        dpg.draw_circle(pos, 4, color=(255,255,0,255), fill=(255,255,0,255), parent="canvas_drawlist")
    # --- Fixed ---
    for f in fixed:
        bar = bars[f[0]]
        p1 = np.array(points[bar[0]])
        p2 = np.array(points[bar[1]])
        c = (p1 + p2)/2
        d = p2 - p1
        pos = c + d*f[1]
        dpg.draw_circle(grid_to_canvas(pos), 4, color=(0,255,255,255), fill=(0,255,255,255), parent="canvas_drawlist")

# --- Reference load callback ---
def load_reference_image(sender, app_data):
    global ref_texture, ref_filename, ref_width, ref_height, ref_offset, ref_scale, ref_alpha

    filename = app_data["file_path_name"]
    # load image from file (returns width, height, channels, and pixel data)
    result = dpg.load_image(filename)
    if result is None:
        print("Failed to load image:", filename)
        return
    width, height, channels, image_data = result

    ref_width = width
    ref_height = height

    # create static texture from image data
    with dpg.texture_registry(show=False):
        ref_texture = dpg.add_static_texture(width, height, image_data)

    ref_offset = np.array([-width / 2, -height / 2])
    ref_scale = 1.0
    ref_alpha = .2

    ref_filename = filename
    dpg.set_value("ref_file_text", f"File: {filename}")
    dpg.set_value("img_offset_x", ref_offset[0])
    dpg.set_value("img_offset_y", ref_offset[1])
    dpg.set_value("img_scale", ref_scale)
    dpg.set_value("img_alpha", ref_alpha)
    draw_canvas()

def clear_reference_image(sender, app_data):
    global ref_texture, ref_filename
    ref_texture = None
    ref_filename = ""
    draw_canvas()

def update_image_params(sender, app_data):
    global ref_offset, ref_scale, ref_alpha
    ref_offset[0] = dpg.get_value("img_offset_x")
    ref_offset[1] = dpg.get_value("img_offset_y")
    ref_scale = dpg.get_value("img_scale")
    ref_alpha = int(dpg.get_value("img_alpha")*255)
    draw_canvas()

# --- Export / Import ---
def export_file(sender, app_data):
    name = "objects/mechanism.txt"
    all_joints = explicit_joints.copy()
    implicit = get_implicit_joints()
    for ij in implicit:
        if sorted(ij) not in [sorted(x) for x in all_joints]:
            all_joints.append(ij)
    with open(name,"w") as f:
        f.write("POINTS\n")
        for p in points:
            f.write(f"{p[0]*grid_scale} {p[1]*grid_scale}\n")
        f.write("BARS\n")
        for b in bars:
            f.write(f"{b[0]} {b[1]}\n")
        f.write("JOINTS\n")
        for j in all_joints:
            f.write(f"{j[0]} {j[1]}\n")
        f.write("FIXED\n")
        for fi in fixed:
            f.write(f"{fi[0]} {fi[1]}\n")
    print(f"Exported {name}")

def load_file(sender, app_data):
    global points, bars, explicit_joints, fixed
    filename = app_data['file_path_name']
    pts, brs, jnts, fxd = [], [], [], []
    mode = None
    with open(filename,"r") as f:
        for line in f:
            line = line.strip()
            if line=="POINTS": mode="p"; continue
            elif line=="BARS": mode="b"; continue
            elif line=="JOINTS": mode="j"; continue
            elif line=="FIXED": mode="f"; continue
            if not line: continue
            vals = list(map(float, line.split()))
            if mode=="p": pts.append([vals[0]/grid_scale, vals[1]/grid_scale])
            elif mode=="b": brs.append([int(vals[0]), int(vals[1])])
            elif mode=="j": jnts.append([int(vals[0]), int(vals[1])])
            elif mode=="f": fxd.append([int(vals[0]), vals[1]])
    points, bars, explicit_joints, fixed = pts, brs, jnts, fxd
    draw_canvas()
    update_data_inspector()

def scale_points(sender, app_data):
    factor = dpg.get_value("scale_input")
    for i in range(len(points)):
        points[i] *= factor
    draw_canvas()
    update_data_inspector()

# --- Mouse callback ---
def canvas_click_callback(sender, app_data):
    global selected_points, selected_bars, bars, explicit_joints, fixed
    pos = get_mouse_pos_on_canvas()
    if pos is None: return
    pos_grid = canvas_to_grid(pos)
    if dpg.is_mouse_button_released(1):
        canvas_right_click_callback(sender, app_data)
        return

    if mode == "points":
        # Only add if no existing point is too close
        if all(np.linalg.norm(pos_grid - np.array(p)) >= .2 for p in points):
            points.append(pos_grid.tolist())
    elif mode == "bars":
        if len(points) == 0: return
        distances = [np.linalg.norm(np.array(p)-pos_grid) for p in points]
        idx = int(np.argmin(distances))
        selected_points.append(idx)
        if len(selected_points)==2:
            p1,p2 = selected_points
            bars.append([p1,p2])
            selected_points=[]
    elif mode=="joints":
        for i,b in enumerate(bars):
            p1,p2=np.array(points[b[0]]),np.array(points[b[1]])
            d=p2-p1
            t=np.clip(np.dot(pos_grid-p1,d)/np.dot(d,d),0,1)
            proj=p1+t*d
            if np.linalg.norm(proj-pos_grid)<0.2:
                if i not in selected_bars: selected_bars.append(i)
                break
        if len(selected_bars)==2:
            b1,b2=selected_bars
            if b1!=b2:
                if len(set(bars[b1])&set(bars[b2]))==0:
                    if sorted([b1,b2]) not in [sorted(x) for x in explicit_joints]:
                        explicit_joints.append([b1,b2])
            selected_bars=[]
    elif mode=="fixed":
        for i,b in enumerate(bars):
            p1,p2=np.array(points[b[0]]),np.array(points[b[1]])
            d=p2-p1
            t=np.clip(np.dot(pos_grid-p1,d)/np.dot(d,d),0,1)
            proj=p1+t*d
            if np.linalg.norm(proj-pos_grid)<0.2:
                if t < 0.2:
                    t = 0.0
                elif t > 0.8:
                    t = 1.0
                fixed.append([i,t-0.5])
                break
    draw_canvas()
    update_data_inspector()

def canvas_right_click_callback(sender, app_data):
    # Simplified: delete nearest element
    pos = get_mouse_pos_on_canvas()
    if pos is None: return
    pos_grid = canvas_to_grid(pos)
    # Points
    for i,p in enumerate(points):
        if np.linalg.norm(np.array(p)-pos_grid)<0.2:
            points.pop(i)
            draw_canvas()
            update_data_inspector()
            return

# --- UI ---
dpg.create_context()

def set_mode(new_mode):
    global mode, selected_points, selected_bars
    mode=new_mode
    selected_points=[]
    selected_bars=[]
    dpg.set_value("mode_text", mode)

with dpg.window(label="Controls", width=250, height=350):
    dpg.add_text("Mode:")
    dpg.add_text(default_value=mode, tag="mode_text")
    dpg.add_button(label="Points", callback=lambda s,a:set_mode("points"))
    dpg.add_button(label="Bars", callback=lambda s,a:set_mode("bars"))
    dpg.add_button(label="Joints", callback=lambda s,a:set_mode("joints"))
    dpg.add_button(label="Fixed", callback=lambda s,a:set_mode("fixed"))
    dpg.add_input_float(label="Scale Factor", default_value=1.0, tag="scale_input")
    dpg.add_button(label="Scale All Points", callback=scale_points)
    dpg.add_button(label="Export", callback=export_file)
    dpg.add_button(label="Load File", callback=lambda s,a:dpg.show_item("load_file_dialog"))

with dpg.file_dialog(label="Load Mechanism", callback=load_file, show=False, tag="load_file_dialog"):
    dpg.add_file_extension(".txt")

with dpg.window(label="Data Inspector", width=250, height=CANVAS_SIZE-250, pos=(0,360)):
    with dpg.collapsing_header(label="Points", default_open=True):
        dpg.add_text("", tag="points_text")
    with dpg.collapsing_header(label="Bars", default_open=True):
        dpg.add_text("", tag="bars_text")
    with dpg.collapsing_header(label="Explicit Joints", default_open=True):
        dpg.add_text("", tag="joints_text")
    with dpg.collapsing_header(label="Fixed Points", default_open=True):
        dpg.add_text("", tag="fixed_text")

def update_data_inspector():
    dpg.set_value("points_text","\n".join([str(p) for p in points]))
    dpg.set_value("bars_text","\n".join([str(b) for b in bars]))
    dpg.set_value("joints_text","\n".join([str(j) for j in explicit_joints]))
    dpg.set_value("fixed_text","\n".join([str(f) for f in fixed]))

# --- Canvas ---
with dpg.window(label="Canvas Window", width=CANVAS_SIZE+20, height=CANVAS_SIZE+20, pos=(260,0)):
    with dpg.drawlist(width=CANVAS_SIZE, height=CANVAS_SIZE, tag="canvas_drawlist"):
        pass

with dpg.window(label="Reference Image Controls", width=300, height=300, pos=(260, CANVAS_SIZE+10)):
    dpg.add_text("File: ", tag="ref_file_text")
    dpg.add_button(label="Load", callback=lambda s,a:dpg.show_item("ref_file_dialog"))
    dpg.add_input_float(label="Offset X", tag="img_offset_x", default_value=0.0, callback=update_image_params)
    dpg.add_input_float(label="Offset Y", tag="img_offset_y", default_value=0.0, callback=update_image_params)
    dpg.add_input_float(label="Scale", tag="img_scale", default_value=1.0, min_value=0.01, callback=update_image_params)
    dpg.add_input_float(label="Alpha (0-1)", tag="img_alpha", default_value=0.5, min_value=0.0, max_value=1.0, callback=update_image_params)
    dpg.add_button(label="Clear Image", callback=clear_reference_image)

with dpg.file_dialog(label="Load Reference Image", callback=load_reference_image, show=False, tag="ref_file_dialog"):
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".jpg")

dpg.set_item_callback("canvas_drawlist", canvas_click_callback)
draw_canvas()
dpg.create_viewport(title='2D Mechanism Editor', width=CANVAS_SIZE+260, height=CANVAS_SIZE)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()