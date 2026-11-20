#!/usr/bin/env python3
"""Offline clean-up / segmentation / meshing of a saved scan with Open3D (run with ~/playground/.venv-o3d/bin/python).

  postprocess.py scan.ply [--max-dist 1.5] [--voxel 0.01] [--planes] [--mesh] [--out DIR]

 1. keeps points acquired from <= --max-dist (PLY `distance` column written by pcserver: stereo noise ~ distance^2)
 2. statistical outlier removal, optional voxel downsample
 3. --planes: iterative RANSAC planes -> labels floor / ceiling / wall / furniture(other horizontal) / other
    (zones.py LABELS ids); the floor is the lowest large horizontal plane facing up, the ceiling the highest facing down.
    Works with a sloped floor if --tilt allows it (default 15 deg from horizontal counts as horizontal).
 4. --mesh: normals + Poisson reconstruction (low-density vertices trimmed) -> OUT/NAME_mesh.ply
Outputs: OUT/NAME_clean.ply (points + label [+ time, distance]) and the mesh. STAGE coordinates (y up) unless the scan was
saved with --origin-frame (see its .ply.json).
"""
import argparse, os, sys
import numpy as np
import open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcpub, zones

ap = argparse.ArgumentParser(); ap.add_argument("ply")
ap.add_argument("--max-dist", type=float); ap.add_argument("--voxel", type=float, default=0.0)
ap.add_argument("--planes", action="store_true"); ap.add_argument("--mesh", action="store_true")
ap.add_argument("--tilt", type=float, default=15.0); ap.add_argument("--plane-tol", type=float, default=0.02)
ap.add_argument("--min-plane", type=int, default=2000); ap.add_argument("--depth", type=int, default=9)
ap.add_argument("--out"); a = ap.parse_args()
out = a.out or os.path.dirname(os.path.abspath(a.ply)); os.makedirs(out, exist_ok=True)
name = os.path.splitext(os.path.basename(a.ply))[0]

P, C, A = pcpub.read_ply(a.ply, attrs=True)
P = P.astype(np.float64); n0 = len(P); keep = np.ones(n0, bool)
if a.max_dist and "distance" in A:
    keep &= A["distance"] <= a.max_dist
elif a.max_dist:
    print("(no distance column in this PLY: --max-dist ignored)")
pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P[keep]); pc.colors = o3d.utility.Vector3dVector(C[keep] / 255.0)
idx = np.flatnonzero(keep)
_, ok = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0); idx = idx[np.asarray(ok)]
pc = pc.select_by_index(ok)
print(f"{n0} points -> {len(idx)} after distance gate + outlier removal")
if a.voxel > 0:                                        # keep attribute columns by nearest surviving point
    pc2 = pc.voxel_down_sample(a.voxel); tree = o3d.geometry.KDTreeFlann(pc)
    sel = np.array([tree.search_knn_vector_3d(p, 1)[1][0] for p in np.asarray(pc2.points)]); idx = idx[sel]; pc = pc2
    print(f"voxel {a.voxel} m: {len(idx)} points")
Pk = np.asarray(pc.points)
lab = np.zeros(len(Pk), np.uint8)

if a.planes:
    L = {n: zones.label_id(n) for n in zones.LABELS}
    rest = np.arange(len(Pk)); planes = []
    cos_h = np.cos(np.radians(a.tilt)); cos_w = np.sin(np.radians(a.tilt))
    for _ in range(12):
        if len(rest) < a.min_plane: break
        sub = pc.select_by_index(rest)
        model, inl = sub.segment_plane(a.plane_tol, 3, 1500)
        if len(inl) < a.min_plane: break
        g = rest[np.asarray(inl)]; nrm = np.array(model[:3]) / np.linalg.norm(model[:3])
        planes.append((g, nrm, float(np.median(Pk[g, 1])))); rest = np.setdiff1d(rest, g)
    ups = [p for p in planes if abs(p[1][1]) > cos_h]
    floor = min((p for p in ups), key=lambda p: p[2], default=None); ceil = max((p for p in ups), key=lambda p: p[2], default=None)
    for g, nrm, y in planes:
        if abs(nrm[1]) > cos_h:
            name_ = "floor" if (floor and g is floor[0]) else "ceiling" if (ceil and g is ceil[0] and ceil is not floor) else "furniture"
        elif abs(nrm[1]) < cos_w: name_ = "wall"
        else: name_ = "other"
        lab[g] = L[name_]; print(f"  plane {name_:9s} {len(g):7d} pts  normal ({nrm[0]:+.2f},{nrm[1]:+.2f},{nrm[2]:+.2f})  y {y:+.2f}")
    print("labelled", int((lab > 0).sum()), "of", len(lab))

cols = {k: v[idx] for k, v in A.items() if k in ("time", "distance")}
pcpub.write_ply(os.path.join(out, name + "_clean.ply"), Pk.astype(np.float32), (np.asarray(pc.colors) * 255).astype(np.uint8),
                lab if a.planes else None, t=cols.get("time"), dist=cols.get("distance"))
print("wrote", os.path.join(out, name + "_clean.ply"))

if a.mesh:
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    pc.orient_normals_consistent_tangent_plane(30)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=a.depth)
    d = np.asarray(dens); mesh.remove_vertices_by_mask(d < np.quantile(d, 0.05))
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    fn = os.path.join(out, name + "_mesh.ply"); o3d.io.write_triangle_mesh(fn, mesh)
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {fn}")
