"""Blender import of a headset session: point cloud + planning objects as editable Blender objects.

  blender --python tools/blender_import.py -- captures/sessions/scan_20261118_101500.json [--enu] [--point-radius 0.004]
  (or in Blender's Scripting tab: set ARGS below and run)

Creates a collection "<scene name>" with
  PointCloud  mesh vertices + colour attribute "Col"; a Geometry Nodes modifier shows them as points
  Box1...     cube meshes with the real dimensions (object rotation/location, scale 1)
  Path1...    poly curves with bevel = conduit diameter (edit the points, change the bevel)
  A1...       empties (arrows); custom properties lat/lon/h when the scene is georeferenced
Axes: headset STAGE (x right, y up, z back) -> Blender (x, -z, y), i.e. Blender Z is up.
--enu (georeferenced scenes): Blender X = East, Y = North, Z = Up around the georef origin.
NOTE: written without a Blender binary at hand (only the glTF path was tested); report issues.
"""
import json, math, os, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def stage_to_blender_matrix(georef=None, enu=False):
    """3x4 affine: blender = R @ stage + t"""
    if enu and georef:
        c, s = math.cos(georef["yaw"]), math.sin(georef["yaw"])
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        M = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])          # stage -> ENU at yaw 0
        return Rz @ M, np.array(georef["t"], float)
    return np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]), np.zeros(3)


def read_ply(fn):
    raw = open(fn, "rb").read(); h = raw.index(b"end_header\n") + 11
    a = np.frombuffer(raw[h:], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return a["p"].astype(float), a["c"]


def points_material():
    m = bpy.data.materials.get("ScanPoints") or bpy.data.materials.new("ScanPoints")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    at = nt.nodes.new("ShaderNodeAttribute"); at.attribute_name = "Col"
    nt.links.new(at.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs[0], out.inputs["Surface"])
    return m


def points_modifier(obj, radius):
    ng = bpy.data.node_groups.new("ScanPointsDisplay", "GeometryNodeTree")
    if hasattr(ng, "interface"):      # Blender 4.x
        ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    else:                             # Blender 3.x
        ng.inputs.new("NodeSocketGeometry", "Geometry"); ng.outputs.new("NodeSocketGeometry", "Geometry")
    gi = ng.nodes.new("NodeGroupInput"); go = ng.nodes.new("NodeGroupOutput")
    mp = ng.nodes.new("GeometryNodeMeshToPoints"); mp.inputs["Radius"].default_value = radius
    sm = ng.nodes.new("GeometryNodeSetMaterial"); sm.inputs["Material"].default_value = points_material()
    ng.links.new(gi.outputs[0], mp.inputs["Mesh"]); ng.links.new(mp.outputs["Points"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    mod = obj.modifiers.new("ShowPoints", "NODES"); mod.node_group = ng


def flat_material(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    if b:
        b.inputs["Base Color"].default_value = rgba
        if "Alpha" in b.inputs:
            b.inputs["Alpha"].default_value = rgba[3]
    if rgba[3] < 1 and hasattr(m, "blend_method"):
        m.blend_method = "BLEND"
    return m


def main(args):
    import argparse
    ap = argparse.ArgumentParser(prog="blender_import.py")
    ap.add_argument("scene"); ap.add_argument("--enu", action="store_true"); ap.add_argument("--point-radius", type=float, default=0.004)
    ap.add_argument("--ply")
    a = ap.parse_args(args)
    sc = json.load(open(a.scene)); name = os.path.splitext(os.path.basename(a.scene))[0]
    g = sc.get("georef") if (sc.get("georef") or {}).get("ok") else None
    R, t = stage_to_blender_matrix(g, a.enu)
    tp = lambda P: np.asarray(P, float) @ R.T + t
    col = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(col)
    if g:
        col["georef"] = json.dumps(g)

    ply = a.ply or (os.path.join(os.path.dirname(os.path.abspath(a.scene)), sc["cloud"]) if sc.get("cloud") else None)
    if ply and os.path.exists(ply):
        P, C = read_ply(ply); P = tp(P)
        me = bpy.data.meshes.new("PointCloud")
        me.vertices.add(len(P)); me.vertices.foreach_set("co", P.astype(np.float32).ravel())
        ca = me.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
        rgba = np.c_[C / 255.0, np.ones(len(C))].astype(np.float32)
        ca.data.foreach_set("color", rgba.ravel())
        me.update()
        ob = bpy.data.objects.new("PointCloud", me); col.objects.link(ob)
        try:
            points_modifier(ob, a.point_radius)
        except Exception as e:   # older/newer node API: points stay visible as vertices in edit mode
            print("point display modifier failed:", e)

    m_box = flat_material("PlanBox", (0.25, 0.75, 1.0, 0.35)); m_path = flat_material("PlanPath", (1.0, 0.55, 0.1, 1.0))
    for o in sc.get("objects", []):
        typ, nm = o["type"], o.get("name", o["type"])
        if typ == "anchor":
            ob = bpy.data.objects.new(nm, None); ob.empty_display_type = "ARROWS"; ob.empty_display_size = 0.1
            ob.location = Vector(tp(o["p"]).tolist())
            if g:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                try:
                    import geo
                    lat, lon, h = geo.stage_to_lla(np.array(o["p"], float), g)
                    ob["lat"], ob["lon"], ob["h"] = float(lat), float(lon), float(h)
                except ImportError:
                    pass
            col.objects.link(ob)
        elif typ == "box":
            sx, sy, sz = [v / 2 for v in o["size"]]
            vs = [(x, y, z) for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]
            fs = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
            me = bpy.data.meshes.new(nm); me.from_pydata(vs, [], fs); me.update()
            me.materials.append(m_box)
            ob = bpy.data.objects.new(nm, me)
            Rb = R @ np.array(o["axes"], float).T
            M = Matrix.Identity(4)
            for i in range(3):
                for j in range(3):
                    M[i][j] = Rb[i, j]
            M.translation = Vector(tp(o["center"]).tolist())
            ob.matrix_world = M
            ob["size_m"] = list(o["size"])
            col.objects.link(ob)
        elif typ == "path":
            pts = tp(o["points"]); d = float(o.get("diameter", 0.02))
            cu = bpy.data.curves.new(nm, "CURVE"); cu.dimensions = "3D"
            sp = cu.splines.new("POLY"); sp.points.add(len(pts) - 1)
            for i, p in enumerate(pts):
                sp.points[i].co = (p[0], p[1], p[2], 1.0)
            cu.bevel_depth = d / 2; cu.bevel_resolution = 3; cu.use_fill_caps = True
            cu.materials.append(m_path)
            ob = bpy.data.objects.new(nm, cu); ob["diameter_m"] = d; col.objects.link(ob)
    print(f"imported {name}: {len(sc.get('objects', []))} objects")


if __name__ == "__main__":
    main(ARGS)
