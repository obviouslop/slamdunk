#!/usr/bin/env python3
"""Export a saved session (viewer scene JSON + point cloud PLY) for Blender / other tools.

  scene_export.py captures/sessions/scan_20261118_101500.json [--ply cloud.ply] [--enu] [--out DIR]

Writes next to the scene (or into --out):
  <name>.glb          glTF 2.0 binary: point cloud (POINTS + vertex colours, unlit), boxes (real-size
                      meshes, named Box1...), paths (tube meshes, Path1..., extras: points, diameter,
                      length), measurements (thin tubes M1..., extras: length, dh, horizontal), anchors (empties A1...). Blender: File > Import > glTF 2.0.
  <name>_anchors.csv  anchors: stage xyz (+ ENU and WGS84 lat/lon/h when georeferenced)
Coordinates: glTF is y-up like the headset's STAGE frame, so by default the export is in STAGE metres.
--enu (needs a georef in the scene, tools/gnssd.py): x = East, y = Up, z = -North around the georef
origin (lat0/lon0/h0 in the glTF asset extras); Blender then shows X = East, Y = North, Z = Up.
For editable curves / empties in Blender see tools/blender_import.py.
"""
import argparse, json, os, struct, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo, pcpub

Q_ENU_TO_GLTF = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])


class Glb:
    def __init__(self):
        self.bin = bytearray(); self.views = []; self.acc = []; self.meshes = []; self.nodes = []; self.mats = []

    def _view(self, data, target=None):
        while len(self.bin) % 4: self.bin += b"\0"
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target: v["target"] = target
        self.bin += data; self.views.append(v); return len(self.views) - 1

    def accessor(self, arr, ctype, typ, normalized=False, target=34962, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype, "count": int(arr.shape[0]), "type": typ}
        if normalized: a["normalized"] = True
        if minmax:
            a["min"] = arr.min(0).astype(float).tolist(); a["max"] = arr.max(0).astype(float).tolist()
        self.acc.append(a); return len(self.acc) - 1

    def material(self, name, rgba, blend=False, unlit=False):
        m = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": list(rgba), "metallicFactor": 0.0, "roughnessFactor": 0.9},
             "doubleSided": True}
        if blend: m["alphaMode"] = "BLEND"
        if unlit: m["extensions"] = {"KHR_materials_unlit": {}}
        self.mats.append(m); return len(self.mats) - 1

    def mesh(self, name, P, idx=None, N=None, C=None, mat=None, mode=4):
        attrs = {"POSITION": self.accessor(P.astype(np.float32), 5126, "VEC3", minmax=True)}
        if N is not None: attrs["NORMAL"] = self.accessor(N.astype(np.float32), 5126, "VEC3")
        if C is not None: attrs["COLOR_0"] = self.accessor(C.astype(np.uint8), 5121, "VEC4", normalized=True)
        prim = {"attributes": attrs, "mode": mode}
        if idx is not None: prim["indices"] = self.accessor(idx.astype(np.uint32).ravel(), 5125, "SCALAR", target=34963)
        if mat is not None: prim["material"] = mat
        self.meshes.append({"name": name, "primitives": [prim]}); return len(self.meshes) - 1

    def node(self, name, mesh=None, t=None, r=None, extras=None):
        n = {"name": name}
        if mesh is not None: n["mesh"] = mesh
        if t is not None: n["translation"] = [float(x) for x in t]
        if r is not None: n["rotation"] = [float(x) for x in r]
        if extras: n["extras"] = extras
        self.nodes.append(n); return len(self.nodes) - 1

    def write(self, fn, root_children, extras=None):
        scene_root = {"name": "Scan", "children": root_children}
        self.nodes.append(scene_root)
        gl = {"asset": {"version": "2.0", "generator": "steam-frame-pointcloud scene_export.py"},
              "extensionsUsed": ["KHR_materials_unlit"], "scene": 0, "scenes": [{"nodes": [len(self.nodes) - 1]}],
              "nodes": self.nodes, "meshes": self.meshes, "materials": self.mats, "accessors": self.acc,
              "bufferViews": self.views, "buffers": [{"byteLength": len(self.bin)}]}
        if extras: gl["asset"]["extras"] = extras
        js = json.dumps(gl, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        while len(self.bin) % 4: self.bin += b"\0"
        with open(fn, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A)); f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942)); f.write(self.bin)


def box_mesh(size):
    h = np.asarray(size, float) / 2
    P, N, I = [], [], []
    for ax in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3); n[ax] = sgn
            u, v = [(a) for a in range(3) if a != ax]
            base = len(P)
            for cu, cv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3); p[ax] = sgn * h[ax]; p[u] = cu * h[u]; p[v] = cv * h[v]; P.append(p); N.append(n)
            tri = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            if np.dot(np.cross(P[base + 1] - P[base], P[base + 2] - P[base]), n) < 0:
                tri = [[t[0], t[2], t[1]] for t in tri]             # counter-clockwise seen from outside
            I += tri
    return np.array(P), np.array(N), np.array(I)


def tube_mesh(pts, r, sides=12):
    P, N, I = [], [], []
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a; L = np.linalg.norm(d)
        if L < 1e-5: continue
        y = d / L; ref = np.array([0, 1., 0]) if abs(y[1]) < 0.9 else np.array([1., 0, 0])
        x = np.cross(y, ref); x /= np.linalg.norm(x); z = np.cross(x, y)
        base = len(P)
        for k in range(sides):
            ang = 2 * np.pi * k / sides; n = np.cos(ang) * x + np.sin(ang) * z
            P += [a + r * n, b + r * n]; N += [n, n]
        for k in range(sides):
            i0, i1 = base + 2 * k, base + 2 * ((k + 1) % sides)
            I += [[i0, i0 + 1, i1], [i1, i0 + 1, i1 + 1]]
    return np.array(P), np.array(N), np.array(I)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene"); ap.add_argument("--ply"); ap.add_argument("--enu", action="store_true")
    ap.add_argument("--out"); ap.add_argument("--max-points", type=int, default=2_000_000)
    a = ap.parse_args()
    sc = json.load(open(a.scene)); base = os.path.splitext(os.path.basename(a.scene))[0]
    outdir = a.out or os.path.dirname(os.path.abspath(a.scene)); os.makedirs(outdir, exist_ok=True)
    ply = a.ply or (os.path.join(os.path.dirname(os.path.abspath(a.scene)), sc["cloud"]) if sc.get("cloud") else None)
    g = sc.get("georef") if sc.get("georef", {}).get("ok") else None
    if a.enu and not g:
        sys.exit("--enu needs a georeference in the scene (georef.ok)")
    if a.enu:
        Rv = Q_ENU_TO_GLTF @ geo.stage_rot_to_enu(g)
        tp = lambda P: geo.stage_to_enu(P, g) @ Q_ENU_TO_GLTF.T
    else:
        Rv = np.eye(3); tp = lambda P: np.asarray(P, float)

    G = Glb(); kids = []
    if ply and os.path.exists(ply):
        P, C = pcpub.read_ply(ply)
        if len(P) > a.max_points:
            P, C = P[:: len(P) // a.max_points + 1], C[:: len(P) // a.max_points + 1]
        C4 = np.c_[C, np.full(len(C), 255, np.uint8)]
        m = G.mesh("PointCloud", tp(P), C=C4, mat=G.material("points", (1, 1, 1, 1), unlit=True), mode=0)
        kids.append(G.node("PointCloud", m)); print(f"cloud: {len(P)} points from {ply}")
    elif ply:
        print(f"warning: cloud {ply} not found, exporting objects only")
    m_box = G.material("box", (0.25, 0.75, 1.0, 0.35), blend=True); m_path = G.material("path", (1.0, 0.55, 0.1, 1.0))
    m_meas = G.material("measure", (1.0, 1.0, 0.3, 1.0))
    rows = []
    for o in sc.get("objects", []):
        t = o["type"]; name = o.get("name", t)
        if t == "anchor":
            p = np.array(o["p"], float); ex = {"stage": o["p"]}
            if g:
                lat, lon, h = geo.stage_to_lla(p, g); enu = geo.stage_to_enu(p, g)
                ex.update(lat=float(lat), lon=float(lon), h=float(h), enu=enu.round(4).tolist())
            kids.append(G.node(name, t=tp(p), extras=ex)); rows.append((name, p, ex))
        elif t == "box":
            ax = np.array(o["axes"], float).T                 # columns = box axes in stage
            R = Rv @ ax
            Pm, Nm, Im = box_mesh(o["size"])
            m = G.mesh(name, Pm, Im, Nm, mat=m_box)
            kids.append(G.node(name, m, t=tp(np.array(o["center"], float)), r=Rot.from_matrix(R).as_quat(),
                               extras={"size_m": o["size"], "points_stage": o.get("points", [])}))
        elif t in ("path", "measure"):
            pts = tp(np.array(o["points"], float)); d = float(o.get("diameter", 0.02 if t == "path" else 0.005))
            Pm, Nm, Im = tube_mesh(pts, d / 2)
            if len(Pm):
                m = G.mesh(name, Pm, Im, Nm, mat=m_path if t == "path" else m_meas)
                L = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                ex = {"type": t, "points": pts.round(5).tolist(), "diameter_m": d, "length_m": round(L, 4)}
                if t == "measure":
                    s = np.array(o["points"], float); ex["dh_m"] = round(float(s[1, 1] - s[0, 1]), 4)
                    ex["horizontal_m"] = round(float(np.hypot(*(s[1] - s[0])[[0, 2]])), 4)
                kids.append(G.node(name, m, extras=ex))
    extras = {"frame": "ENU (x=E, y=U, z=-N)" if a.enu else "STAGE (OpenXR) metres, y up", "source": os.path.basename(a.scene)}
    if g: extras["georef"] = g
    fn = os.path.join(outdir, base + ("_enu" if a.enu else "") + ".glb")
    G.write(fn, kids, extras)
    print(f"wrote {fn} ({os.path.getsize(fn)/1e6:.1f} MB, {len(sc.get('objects', []))} objects)")
    if rows:
        cf = os.path.join(outdir, base + "_anchors.csv")
        with open(cf, "w") as f:
            f.write("name,x,y,z,east,north,up,lat,lon,h\n")
            for name, p, ex in rows:
                e = ex.get("enu", ["", "", ""])
                f.write(f"{name},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{e[0]},{e[1]},{e[2]},{ex.get('lat','')},{ex.get('lon','')},{ex.get('h','')}\n")
        print(f"wrote {cf}")


if __name__ == "__main__":
    main()
