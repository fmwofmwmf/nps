from pxr import Usd, UsdGeom, Gf, UsdShade
import numpy as np


def save_frames_universal(save_path, record_frames, system, fps=24):
    """
    Save recorded frames to USD format.

    Handles both mesh-based systems and point-based systems (pendulums, springs, etc.)

    Args:
        save_path: Output .usda file path
        record_frames: List of frames, where each frame is a list of body_data dicts
        system: The physics system object (Pendulum2DSystem, MassSpring2DSystem, etc.)
        fps: Frames per second for playback

    Body data format (from visualize with return_transforms=True):
        For pendulums:
            {'name': str, 'start': [x,y], 'end': [x,y], 'length': float, 'mass': float}
        For mass-spring:
            {'name': str, 'position': [x,y], 'mass': float}
        For meshes:
            {'vertices': array, 'faces': array}
    """
    stage = Usd.Stage.CreateNew(save_path)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(record_frames) - 1)

    # Set up axis (Blender uses Z-up)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")

    num_frames = len(record_frames)
    if num_frames == 0:
        print("No frames to export!")
        return

    num_bodies = len(record_frames[0])

    # Detect data type from first frame
    first_body = record_frames[0][0]

    if 'vertices' in first_body and 'faces' in first_body:
        # Mesh-based system
        _save_mesh_bodies(stage, record_frames, num_bodies)
    elif 'start' in first_body and 'end' in first_body:
        # Pendulum system (links with start/end points)
        _save_pendulum_bodies(stage, record_frames, num_bodies)
    elif 'position' in first_body:
        # Point mass system (springs, particles, etc.)
        # Check if system has springs to export
        if hasattr(system, 'spring_connections') and hasattr(system, 'spring_stiffness'):
            spring_info = {
                'connections': system.spring_connections,
                'stiffness': system.spring_stiffness.cpu().numpy().tolist() if hasattr(system.spring_stiffness,
                                                                                       'cpu') else system.spring_stiffness.tolist()
            }
            _save_point_bodies_with_springs(stage, record_frames, num_bodies, spring_info)
        else:
            _save_point_bodies(stage, record_frames, num_bodies)
    else:
        raise ValueError(f"Unknown body data format. Keys: {first_body.keys()}")

    stage.GetRootLayer().Save()
    print(f"Saved {num_frames} frames with {num_bodies} bodies to {save_path}")


def _save_mesh_bodies(stage, record_frames, num_bodies):
    """Handle mesh-based bodies (original code)"""
    body_meshes = []
    for bid in range(num_bodies):
        body_path = f"/World/Body{bid}"
        body_xform = UsdGeom.Xform.Define(stage, body_path)

        mesh = UsdGeom.Mesh.Define(stage, body_path + "/Mesh")
        mesh.GetSubdivisionSchemeAttr().Set("none")

        # Set faces once (assuming topology doesn't change)
        f = record_frames[0][bid]['faces']
        mesh.GetFaceVertexCountsAttr().Set([len(face) for face in f])
        mesh.GetFaceVertexIndicesAttr().Set(f.flatten().tolist())

        body_meshes.append(mesh)

    # Animate vertices per frame
    for frame_idx, frame in enumerate(record_frames):
        time = float(frame_idx)
        for bid, body_data in enumerate(frame):
            mesh = body_meshes[bid]
            v = body_data['vertices']
            # Convert to list of Vec3d
            points = [Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])) for p in v]
            points_attr = mesh.GetPointsAttr()
            points_attr.Set(points, time=time)


def _save_pendulum_bodies(stage, record_frames, num_bodies):
    """Handle pendulum bodies (links with start/end points)"""
    body_curves = []
    body_translate_ops = []

    for bid in range(num_bodies):
        body_name = record_frames[0][bid].get('name', f'pendulum{bid}')

        # Create a curve for the pendulum link
        curve_path = f"/World/{body_name}_link"
        curve = UsdGeom.BasisCurves.Define(stage, curve_path)
        curve.GetTypeAttr().Set("linear")
        curve.GetWrapAttr().Set("nonperiodic")
        curve.GetCurveVertexCountsAttr().Set([2])  # Each curve has 2 points

        # Create a sphere for the mass at the end
        sphere_path = f"/World/{body_name}_mass"
        sphere_xform = UsdGeom.Xform.Define(stage, sphere_path)
        sphere = UsdGeom.Sphere.Define(stage, sphere_path + "/Sphere")

        # Set radius based on mass (from first frame)
        mass = record_frames[0][bid].get('mass', 1.0)
        radius = 0.03 * np.sqrt(mass)
        sphere.GetRadiusAttr().Set(radius)

        # Create translate op ONCE
        translate_op = sphere_xform.AddTranslateOp()

        body_curves.append(curve)
        body_translate_ops.append(translate_op)

    # Animate positions per frame
    for frame_idx, frame in enumerate(record_frames):
        time = float(frame_idx)
        for bid, body_data in enumerate(frame):
            # Animate curve (link)
            start = np.array(body_data['start'])
            end = np.array(body_data['end'])

            # Convert 2D to 3D
            start_3d = Gf.Vec3d(float(start[0]), float(start[1]), 0.0)
            end_3d = Gf.Vec3d(float(end[0]), float(end[1]), 0.0)

            curve = body_curves[bid]
            points_attr = curve.GetPointsAttr()
            points_attr.Set([start_3d, end_3d], time=time)

            # Animate sphere (mass at end)
            translate_op = body_translate_ops[bid]
            translate_op.Set(end_3d, time=time)


def _save_point_bodies(stage, record_frames, num_bodies):
    """Handle point mass bodies (mass-spring systems, particles)"""
    body_data_list = []

    for bid in range(num_bodies):
        body_name = record_frames[0][bid].get('name', f'mass{bid}')

        # Create sphere for mass
        sphere_path = f"/World/{body_name}"
        sphere_xform = UsdGeom.Xform.Define(stage, sphere_path)
        sphere = UsdGeom.Sphere.Define(stage, sphere_path + "/Sphere")

        # Set radius based on mass (from first frame)
        mass = record_frames[0][bid].get('mass', 1.0)
        radius = 0.05 * np.sqrt(mass)
        sphere.GetRadiusAttr().Set(radius)

        # Add material for better Blender compatibility
        _add_simple_material(stage, sphere.GetPrim(), Gf.Vec3f(0.3, 0.6, 0.9))

        # Create translate op ONCE
        translate_op = sphere_xform.AddTranslateOp()

        body_data_list.append(translate_op)

    # Animate positions per frame
    for frame_idx, frame in enumerate(record_frames):
        time = float(frame_idx)
        for bid, body_data in enumerate(frame):
            position = np.array(body_data['position'])

            # Convert 2D to 3D (swap Y and Z for Blender Z-up)
            pos_3d = Gf.Vec3d(float(position[0]), 0.0, float(position[1]))

            translate_op = body_data_list[bid]
            translate_op.Set(pos_3d, time=time)


def _save_point_bodies_with_springs(stage, record_frames, num_bodies, spring_data):
    """Handle point mass bodies with springs"""
    # Export masses
    body_translate_ops = []
    for bid in range(num_bodies):
        body_name = record_frames[0][bid].get('name', f'mass{bid}')

        sphere_path = f"/World/{body_name}"
        sphere_xform = UsdGeom.Xform.Define(stage, sphere_path)
        sphere = UsdGeom.Sphere.Define(stage, sphere_path + "/Sphere")

        mass = record_frames[0][bid].get('mass', 1.0)
        radius = 0.05 * np.sqrt(mass)
        sphere.GetRadiusAttr().Set(radius)

        # Add material for better Blender compatibility
        _add_simple_material(stage, sphere.GetPrim(), Gf.Vec3f(0.8, 0.3, 0.3))

        # Create translate op ONCE
        translate_op = sphere_xform.AddTranslateOp()
        body_translate_ops.append(translate_op)

    # Export springs as meshes (Blender doesn't handle BasisCurves well in animation)
    spring_meshes = []
    connections = spring_data['connections']

    for spring_idx, (i, j) in enumerate(connections):
        if j < 0:
            continue  # Skip fixed point springs for now

        mesh_path = f"/World/spring{spring_idx}"
        mesh_xform = UsdGeom.Xform.Define(stage, mesh_path)
        mesh = UsdGeom.Mesh.Define(stage, mesh_path + "/Mesh")

        # Create a thin cylinder mesh for the spring
        # 2 vertices per spring (start and end)
        mesh.GetFaceVertexCountsAttr().Set([])  # No faces, just edges
        mesh.GetFaceVertexIndicesAttr().Set([])

        # Create as a curve network instead
        curve_path = f"/World/spring{spring_idx}_curve"
        curve = UsdGeom.BasisCurves.Define(stage, curve_path)
        curve.GetTypeAttr().Set(UsdGeom.Tokens.linear)
        curve.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
        curve.GetCurveVertexCountsAttr().Set([2])
        curve.GetWidthsAttr().Set([0.02])

        # Color by stiffness
        if 'stiffness' in spring_data:
            k = spring_data['stiffness'][spring_idx]
            k_norm = min(k / 50.0, 1.0)
            color = Gf.Vec3f(k_norm, 0.5, 1.0 - k_norm)
            curve.GetDisplayColorAttr().Set([color])

        spring_meshes.append((curve, i, j))

    # Animate everything
    for frame_idx, frame in enumerate(record_frames):
        time = float(frame_idx)

        # Animate masses
        for bid, body_data in enumerate(frame):
            position = np.array(body_data['position'])
            # Convert 2D to 3D (swap Y and Z for Blender Z-up)
            pos_3d = Gf.Vec3d(float(position[0]), 0.0, float(position[1]))

            translate_op = body_translate_ops[bid]
            translate_op.Set(pos_3d, time=time)

        # Animate springs
        for curve, i, j in spring_meshes:
            pos_i = np.array(frame[i]['position'])
            pos_j = np.array(frame[j]['position'])

            # Convert 2D to 3D (swap Y and Z for Blender Z-up)
            pos_i_3d = Gf.Vec3d(float(pos_i[0]), 0.0, float(pos_i[1]))
            pos_j_3d = Gf.Vec3d(float(pos_j[0]), 0.0, float(pos_j[1]))

            points_attr = curve.GetPointsAttr()
            points_attr.Set([pos_i_3d, pos_j_3d], time=time)


def _add_simple_material(stage, prim, color):
    """Add a simple material for Blender compatibility"""
    try:
        material_path = prim.GetPath().AppendChild("material")
        material = UsdShade.Material.Define(stage, material_path)

        shader = UsdShade.Shader.Define(stage, material_path.AppendChild("shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Gf.Vec3f).Set(color)

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(material)
    except:
        pass  # Material creation is optional


def save_frames_with_springs(save_path, record_frames, spring_data=None, fps=24):
    """
    DEPRECATED: Use save_frames_universal() instead, which auto-detects springs.

    This function is kept for backward compatibility.
    """
    print("Warning: save_frames_with_springs is deprecated. Use save_frames_universal() instead.")

    # Just call the universal version
    # Note: This won't work perfectly without the system object, but provides fallback
    stage = Usd.Stage.CreateNew(save_path)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(record_frames) - 1)

    world = UsdGeom.Xform.Define(stage, "/World")

    num_frames = len(record_frames)
    num_bodies = len(record_frames[0])

    if spring_data is not None:
        _save_point_bodies_with_springs(stage, record_frames, num_bodies, spring_data)
    else:
        _save_point_bodies(stage, record_frames, num_bodies)

    stage.GetRootLayer().Save()
    print(f"Saved {num_frames} frames with {num_bodies} masses to {save_path}")
