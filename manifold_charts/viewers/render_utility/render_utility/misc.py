import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../deps/BlenderToolbox/')))
import BlenderToolBox as bt
import math
import bmesh
import bpy
from .definitions import *


# this makes a scene with cameras, background, ground floor, lights, etc. We should always initialize scenes like this, so that they are consistent.
def setup_scene(cam_location,cam_rotations,focal_length = 50, shadowcatcher_height = -0.25, fast = False):

    div = 1
    if fast:
        div = 10 # render it faster for iterating
    imgRes_x = 1080
    imgRes_y = 1080

    # for debug
    # imgRes_x = 200
    # imgRes_y = 200

    numSamples = 1000//div 
    exposure = 1.25 


    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    # set invisible plane (shadow catcher)
    if shadowcatcher_height is not None:
        bt.invisibleGround(shadowBrightness=0.9,location=(0,0,shadowcatcher_height))

    bpy.context.view_layer.cycles.use_denoising = True

    cam = bt.setCamera_from_UI(cam_location, cam_rotations, focalLength = focal_length)

    ## set light
    ## Option1: Three Point Light System 
    sun = None
    bt.setLight_threePoints(radius=2, height=3, intensity=1500, softness=6, keyLoc='left')

    ## set ambient light
    bt.setLight_ambient(color=(0.5,0.5,0.5,1)) 

    ## set gray shadow to completely white with a threshold (optional but recommended)
    bt.shadowThreshold(alphaThreshold = 0.2, interpolationMode = 'CARDINAL')
    return cam, sun

def plot_mesh_with_value(mesh_path, scalar_path, rotation=(90,0,0), scale=(1,1,1), location=(0,0,0),  color_map_name='RdBu', subdiv_level = 2, max_value = float('inf') ):
    mesh = bt.readMesh(mesh_path, location=location, rotation_euler=rotation, scale=scale)

    with open(scalar_path, 'r') as file:
        values = [float(line.strip()) for line in file]
    
    values = np.clip(np.array(values), 0, max_value)

    vertex_colors = bt.colorMap(values, color_map_name)
    mesh = bt.setMeshColors(mesh, vertex_colors, "vertex")

    ## add color to point cloud
    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, meshVColor)

    bt.subdivision(mesh, level = subdiv_level)

    return mesh

# this draws the background mesh on which we plot the curves
def plot_background_mesh(path_to_mesh, color=input_mesh_color,
    location=(0,0,0), rotation=(90,0,0), scale=(1,1,1), edge_thickness=0, edge_color = default_edge_color, subdiv_level = 2):

    mesh = bt.readMesh(path_to_mesh, location, rotation, scale=scale)

    if edge_thickness>0:
        edgeColor = bt.colorObj(edge_color,0.5, 1.0, 1.0, 0.0, 0.0)
        bt.setMat_edge(mesh, edge_thickness, edgeColor, color, 1.0)
        subdiv_level = 0
    else:
        meshcolor = bt.colorObj(color, 0.5, 1.3, 1.0, 0.0, 0.0)
        bt.setMat_plastic(mesh, meshcolor)
    bt.subdivision(mesh, level = subdiv_level)
    
    return mesh

# this draws a curve on top of a mesh.
def draw_curve(path_to_curve,name="cam_path", r=0.04, color=our_color, location=(0,0,0), rotation=(90,0,0), scale=(1,1,1), subdivision_level = 2):
    
    bdColor = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 1)
    bdMesh = bt.readMesh(path_to_curve, location, rotation, scale)
    mesh_data = bdMesh.data

    # Create a new object with the mesh data
    bdObj = bpy.data.objects.new("obj" + name, mesh_data)
    # bdObj.rotation_euler = (90, 0, 0)
    bpy.context.scene.collection.objects.link(bdObj)

    # bevel with a circle
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bdObj
    bdObj.select_set(state=True)
    bpy.ops.object.convert(target='CURVE')
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=(1e5, 1e5, 1e5))
    circ = bpy.context.object
    bdObj.data.bevel_object = circ
    bpy.ops.object.shade_smooth()

    # # subdivision
    if subdivision_level > 0:
        level = subdivision_level
        bpy.context.view_layer.objects.active = bdObj
        bpy.ops.object.modifier_add(type='SUBSURF')
        bdObj.modifiers["Subdivision"].render_levels = level
        bdObj.modifiers["Subdivision"].levels = level 

    bdObj.data.bevel_depth = r

    # add material
    mat = bpy.data.materials.new('MeshMaterial')
    bdObj.data.materials.append(mat)
    bdObj.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    BCNode = bt.initColorNode(tree, bdColor)

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
    tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = bdColor.RGBA[3]
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    bdObj.location = location
    # covert rotation from degree to radian
    rotation = [math.radians(i) for i in rotation]
    bdObj.rotation_euler = rotation
    bdObj.scale = scale

    return bdObj


# This draws OUR mesh from OUR method. We will only ever render our result with this function
def draw_our_curve(path_to_curve,name="cam_path", r=0.04, color=our_color, location=(0,0,0), rotation=(90,0,0), scale=(1,1,1)):
    return draw_curve(path_to_curve, name=name, r=r, color=color, location=location, rotation=rotation, scale=scale)

# This draws the repulsive curves result, for when we compare.
def draw_repulsive_curve(path_to_curve,name="cam_path", r=0.04, color=repulsive_curves_color, location=(0,0,0), rotation=(90,0,0), scale=(1,1,1)):
    return draw_curve(path_to_curve, name=name, r=r, color=color, location=location, rotation=rotation, scale=scale)

def plot_point_cloud(path,r=0.003,color=point_cloud_default_color,rotation=(90,0,0), scale=(1,1,1), location=(0,0,0)):
    mesh = bt.readMesh(path, location=location, rotation_euler=rotation, scale=scale)
    # # set material 
    ptColor = bt.colorObj(color, 0.5, 1.3, 1.0, 0.0, 0.0)
    bt.setMat_pointCloud(mesh, ptColor, r)
    return mesh

def plot_point_cloud_with_value(mesh_path, scalar_path, r=0.003,rotation=(90,0,0), scale=(1,1,1), location=(0,0,0),  color_map_name='RdBu'):
    mesh = bt.readMesh(mesh_path, location=location, rotation_euler=rotation, scale=scale)

    with open(scalar_path, 'r') as file:
        values = [float(line.strip()) for line in file]

    vertex_colors = bt.colorMap(values, color_map_name)

    ## add color to point cloud
    PC = np.array(vertex_colors) # point colors
    mesh = bt.setPointColors(mesh, PC)

    ## set material ptColor = (vertex_RGBA, H, S, V, Bright, Contrast)
    ptColor = bt.colorObj([], 0.5, 1.3, 1.0, 0.0, 0.0)
    bt.setMat_pointCloudColored(mesh, ptColor, r)

    return mesh

def plot_point_cloud_with_color(mesh_path, colors, r=0.003,rotation=(90,0,0), scale=(1,1,1), location=(0,0,0)):
    mesh = bt.readMesh(mesh_path, location=location, rotation_euler=rotation, scale=scale)

    mesh = bt.setPointColors(mesh, colors)

    ## set material ptColor = (vertex_RGBA, H, S, V, Bright, Contrast)
    ptColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 1)
    bt.setMat_pointCloudColored(mesh, ptColor, r)

    return mesh

def plot_vector_field(path_to_obj_with_origins, path_to_obj_with_directions, location=(0,0,0), rotation=(90,0,0), scale=(1,1,1), thickness = 0.01, length = 0.02):
    # read origins and directions into numpy arrays
    bdMesh = bt.readMesh(path_to_obj_with_origins, location, rotation, scale)
    mesh_data = bdMesh.data
    bdMesh.select_set(state=True)
    bpy.ops.object.delete()
    origins = np.array([v.co for v in mesh_data.vertices])

    bdMesh = bt.readMesh(path_to_obj_with_directions, location, rotation, scale)
    mesh_data = bdMesh.data
    bdMesh.select_set(state=True)
    bpy.ops.object.delete()
    directions = np.array([v.co for v in mesh_data.vertices])

    mesh = bt.createVectorFieldMesh(origins, directions, thickness, length, location, rotation, scale)
    meshColor = bt.colorObj(vector_field_default_color, 0.5, 1.0, 1.0, 0.0, 0.5)
    bt.setMat_plastic(mesh, meshColor)
    return mesh