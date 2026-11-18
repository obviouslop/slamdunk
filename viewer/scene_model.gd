class_name SceneModel
extends Node3D
## Planning objects placed in the headset: anchors (named points), boxes (shelves, cabinets, solar
## panels...), paths (wires, conduits), measures and zones (include / exclude / label regions of the cloud,
## see tools/zones.py for the shapes). Coordinates: OpenXR STAGE == OpenVR Standing (metres, y up).
## Serialised as JSON (format "pcscene/1"), see tools/scene_export.py for the glTF / Blender export.

signal changed

const FORMAT := "pcscene/1"
const SNAP_DEG := 7.0          # snap box edges/faces to horizontal/vertical within this angle
const PANEL_THICKNESS := 0.04  # box finished after 3 points (grip) = panel of this thickness

var objects: Array = []        # Dictionaries (serialisable fields only) ...
var nodes: Array = []          # ... and their visual nodes (same index)
var counters := {"anchor": 0, "box": 0, "path": 0, "measure": 0, "zone": 0}
var preview: Node3D
var mat_box: StandardMaterial3D
var mat_edge: StandardMaterial3D
var mat_path: StandardMaterial3D
var mat_anchor: StandardMaterial3D
var mat_preview: StandardMaterial3D
var mat_measure: StandardMaterial3D
var selection: MultiMeshInstance3D   # SELECT brush (pending, not an object)

const LABELS := ["floor", "wall", "ceiling", "furniture", "object", "other"]	# == tools/zones.py
const ZONE_COLS := {"include": Color(0.2, 1.0, 0.35), "exclude": Color(1.0, 0.25, 0.2),
	"floor": Color(0.85, 0.65, 0.3), "wall": Color(0.95, 0.95, 0.75), "ceiling": Color(0.6, 0.8, 1.0),
	"furniture": Color(0.8, 0.4, 1.0), "object": Color(1.0, 0.8, 0.1), "other": Color(0.6, 0.6, 0.6)}
const ZONE_VIS_H := 2.6		# drawn height of floor outlines


func _init() -> void:
	mat_box = _mat(Color(0.25, 0.75, 1.0, 0.22))
	mat_edge = _mat(Color(0.4, 0.9, 1.0, 0.95))
	mat_path = _mat(Color(1.0, 0.55, 0.1, 0.75))
	mat_anchor = _mat(Color(1.0, 0.3, 0.8, 0.95))
	mat_preview = _mat(Color(1.0, 1.0, 0.3, 0.35))
	mat_measure = _mat(Color(1.0, 1.0, 0.4, 0.95))
	preview = Node3D.new()
	add_child(preview)
	selection = MultiMeshInstance3D.new()
	selection.multimesh = _sphere_multimesh()
	selection.material_override = _mat(Color(0.3, 0.8, 1.0, 0.28))
	add_child(selection)


static func _mat(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = c
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	if c.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return m


# ---------------------------------------------------------------- geometry

## Box from clicked points: p0-p1 = first edge, p2 = extent of the second edge (in the plane p0 p1 p2),
## p3 = height along the face normal (missing -> panel of PANEL_THICKNESS). Edges/faces near horizontal
## or vertical are snapped. Returns {center, axes [x, y, z], size} or {} if degenerate.
static func box_from_points(pts: Array, thickness := PANEL_THICKNESS) -> Dictionary:
	if pts.size() < 3:
		return {}
	var p0: Vector3 = pts[0]
	var e1: Vector3 = pts[1] - p0
	var snap := sin(deg_to_rad(SNAP_DEG))
	if abs(e1.normalized().y) < snap:
		e1.y = 0.0                                   # first edge (nearly) level -> level
	elif Vector2(e1.x, e1.z).length() < snap * e1.length():
		e1 = Vector3(0, e1.y, 0)                     # (nearly) plumb -> plumb
	if e1.length() < 0.005:
		return {}
	var x := e1.normalized()
	var v2: Vector3 = pts[2] - p0
	var e2 := v2 - x * v2.dot(x)
	var n := x.cross(e2).normalized()
	if n.length() < 0.5:
		return {}
	# snap the face: horizontal face (normal ~ up) or vertical face (normal ~ level)
	if abs(n.y) > cos(deg_to_rad(SNAP_DEG)) and abs(x.y) < 1e-6:
		n = Vector3(0, sign(n.y), 0)
	elif abs(n.y) < snap:
		n = Vector3(n.x, 0, n.z).normalized()
		if abs(n.dot(x)) > 1e-3:                     # keep orthogonal to x
			n = (n - x * n.dot(x)).normalized()
	var y := n.cross(x).normalized()               # in-plane, perpendicular to the first edge
	var w := v2.dot(y)
	var h := thickness
	if pts.size() >= 4:
		h = (pts[3] - p0).dot(n)
	var hs: float = h if abs(h) > 0.002 else thickness
	var size := Vector3(e1.length(), abs(w), abs(hs))
	var center := p0 + x * e1.length() * 0.5 + y * w * 0.5 + n * hs * 0.5
	# axes: x = first edge, z = face normal (extrusion), y completes a right-handed frame
	var zax := n * (1.0 if hs >= 0 else -1.0)
	var yax := zax.cross(x).normalized()
	return {"center": center, "axes": [x, yax, zax], "size": Vector3(size.x, size.y, size.z)}


static func path_length(pts: Array) -> float:
	var l := 0.0
	for i in range(1, pts.size()):
		l += (pts[i] - pts[i - 1]).length()
	return l


# ---------------------------------------------------------------- objects

func add_anchor(p: Vector3) -> Dictionary:
	counters["anchor"] += 1
	return _add({"type": "anchor", "name": "A%d" % counters["anchor"], "p": p})


func add_box(pts: Array, thickness := PANEL_THICKNESS) -> Dictionary:
	var b := box_from_points(pts, thickness)
	if b.is_empty():
		return {}
	counters["box"] += 1
	b["type"] = "box"
	b["name"] = "Box%d" % counters["box"]
	b["points"] = pts.duplicate()
	return _add(b)


func add_path(pts: Array, diameter := 0.02) -> Dictionary:
	if pts.size() < 2:
		return {}
	counters["path"] += 1
	return _add({"type": "path", "name": "Path%d" % counters["path"], "points": pts.duplicate(), "diameter": diameter})


func add_measure(a: Vector3, b: Vector3) -> Dictionary:
	## Tape measure: 2 points; label shows the distance, height difference and horizontal distance.
	if a.distance_to(b) < 1e-3:
		return {}
	counters["measure"] += 1
	return _add({"type": "measure", "name": "M%d" % counters["measure"], "points": [a, b]})


func add_zone(z: Dictionary) -> Dictionary:
	## z: {mode: include|exclude|label, [label], shape: box|prism|spheres|cut, + shape fields (tools/zones.py)}
	counters["zone"] += 1
	var o := z.duplicate()
	o["type"] = "zone"
	o["name"] = "Z%d" % counters["zone"]
	return _add(o)


func zones() -> Array:
	## Serialisable zone list for /dev/shm/pczones.json (same encoding as to_dict)
	var out := []
	for d in to_dict()["objects"]:
		if d["type"] == "zone":
			out.append(d)
	return out


static func zone_text(o: Dictionary) -> String:
	var m: String = o.get("mode", "exclude")
	var what: String = String(o.get("label", "")).to_upper() if m == "label" else m.to_upper()
	match o.get("shape", ""):
		"cut":
			return "%s CUT %s %.2f m" % [o.get("name", ""), String(o.get("side", "below")).to_upper(), float(o.get("y", 0.0))]
		"spheres":
			return "%s %s (%d dabs)" % [o.get("name", ""), what, (o.get("centers", []) as Array).size()]
		"prism":
			return "%s %s outline" % [o.get("name", ""), what]
	return "%s %s box" % [o.get("name", ""), what]


static func zone_color(o: Dictionary) -> Color:
	var m: String = o.get("mode", "exclude")
	return ZONE_COLS.get(o.get("label", "other") if m == "label" else m, Color(1, 1, 1))


static func _sphere_multimesh() -> MultiMesh:
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	var s := SphereMesh.new()
	s.radius = 1.0
	s.height = 2.0
	s.radial_segments = 10
	s.rings = 5
	mm.mesh = s
	return mm


static func _fill_spheres(mm: MultiMesh, centers: Array, radii: Array) -> void:
	mm.instance_count = centers.size()
	for i in range(centers.size()):
		var r: float = radii[i] if i < radii.size() else 0.05
		mm.set_instance_transform(i, Transform3D(Basis.from_scale(Vector3.ONE * r), centers[i]))


func set_zones_visible(on: bool) -> void:
	for i in range(objects.size()):
		if objects[i]["type"] == "zone":
			nodes[i].visible = on


func set_selection(centers: Array, radii: Array) -> void:
	_fill_spheres(selection.multimesh, centers, radii)


static func measure_text(a: Vector3, b: Vector3) -> String:
	var d := b - a
	return "%.3f m\n(dh %+.3f  horiz %.3f)" % [d.length(), d.y, Vector2(d.x, d.z).length()]


func remove_last() -> Dictionary:
	if objects.is_empty():
		return {}
	var o: Dictionary = objects.pop_back()
	var n: Node3D = nodes.pop_back()
	n.queue_free()
	changed.emit()
	return o


func clear_all() -> void:
	for n in nodes:
		n.queue_free()
	objects.clear()
	nodes.clear()
	counters = {"anchor": 0, "box": 0, "path": 0, "measure": 0, "zone": 0}
	changed.emit()


func _add(o: Dictionary) -> Dictionary:
	objects.append(o)
	var n := _build(o, false)
	add_child(n)
	nodes.append(n)
	changed.emit()
	return o


# ---------------------------------------------------------------- visuals

func _build(o: Dictionary, is_preview: bool) -> Node3D:
	var root := Node3D.new()
	match o["type"]:
		"anchor":
			_build_anchor(root, o)
		"box":
			_build_box(root, o, is_preview)
		"path":
			_build_path(root, o["points"], o.get("diameter", 0.02), is_preview)
		"measure":
			_build_measure(root, o["points"][0], o["points"][1], is_preview)
		"zone":
			_build_zone(root, o)
	return root


func _build_zone(root: Node3D, o: Dictionary) -> void:
	var c := zone_color(o)
	var fill := _mat(Color(c.r, c.g, c.b, 0.12))
	var edge := _mat(Color(c.r, c.g, c.b, 0.9))
	var label_pos := Vector3.ZERO
	match o.get("shape", ""):
		"box":
			var t := _box_transform(o)
			var sz: Vector3 = o["size"]
			var mi := MeshInstance3D.new()
			var bm := BoxMesh.new()
			bm.size = sz
			mi.mesh = bm
			mi.transform = t
			mi.material_override = fill
			root.add_child(mi)
			var h := sz * 0.5
			var cs := []
			for i in range(8):
				cs.append(t * Vector3(h.x * (1 if i & 1 else -1), h.y * (1 if i & 2 else -1), h.z * (1 if i & 4 else -1)))
			var segs := []
			for e in [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]:
				segs.append_array([cs[e[0]], cs[e[1]]])
			_lines(root, segs, edge)
			label_pos = t.origin + Vector3(0, sz.length() * 0.5 + 0.05, 0)
		"prism":
			var pts: Array = o["points"]
			if pts.size() < 2:
				return
			var y0 := INF
			for p in pts:
				y0 = minf(y0, p.y)
			y0 = maxf(y0, float(o.get("ymin", -1e9)))
			var y1 := minf(y0 + ZONE_VIS_H, float(o.get("ymax", 1e9)))
			var segs := []
			for i in range(pts.size()):
				var a: Vector3 = pts[i]
				var b: Vector3 = pts[(i + 1) % pts.size()]
				for y in [y0, y1]:
					segs.append_array([Vector3(a.x, y, a.z), Vector3(b.x, y, b.z)])
				segs.append_array([Vector3(a.x, y0, a.z), Vector3(a.x, y1, a.z)])
				label_pos += Vector3(a.x, 0, a.z) / pts.size()
			_lines(root, segs, edge)
			label_pos.y = y0 + 1.2
		"spheres":
			var mmi := MultiMeshInstance3D.new()
			mmi.multimesh = _sphere_multimesh()
			_fill_spheres(mmi.multimesh, o["centers"], o.get("radii", []))
			mmi.material_override = _mat(Color(c.r, c.g, c.b, 0.18))
			root.add_child(mmi)
			var cs: Array = o["centers"]
			if not cs.is_empty():
				label_pos = cs[cs.size() - 1] + Vector3(0, 0.12, 0)
		"cut":
			var ctr: Vector3 = o.get("c", Vector3.ZERO)
			var y: float = o.get("y", 0.0)
			var segs := []
			for k in range(-4, 5):					 # 4 x 4 m grid at the cut height
				segs.append_array([Vector3(ctr.x + k * 0.5, y, ctr.z - 2), Vector3(ctr.x + k * 0.5, y, ctr.z + 2)])
				segs.append_array([Vector3(ctr.x - 2, y, ctr.z + k * 0.5), Vector3(ctr.x + 2, y, ctr.z + k * 0.5)])
			_lines(root, segs, edge)
			label_pos = Vector3(ctr.x, y + 0.08, ctr.z)
	_label(root, zone_text(o), label_pos, c)


func _lines(root: Node3D, segs: Array, m: Material) -> void:
	if segs.is_empty():
		return
	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_LINES, m)
	for v in segs:
		im.surface_add_vertex(v)
	im.surface_end()
	var mi := MeshInstance3D.new()
	mi.mesh = im
	root.add_child(mi)




func _build_measure(root: Node3D, a: Vector3, b: Vector3, is_preview: bool) -> void:
	var m := mat_preview if is_preview else mat_measure
	_build_segment(root, a, b, 0.0025, m)
	_sphere(root, a, 0.006, m)
	_sphere(root, b, 0.006, m)
	_label(root, measure_text(a, b), (a + b) * 0.5 + Vector3(0, 0.05, 0), Color(1.0, 1.0, 0.5))


func _build_segment(root: Node3D, a: Vector3, b: Vector3, r: float, m: Material) -> void:
	var d := b - a
	if d.length() < 1e-4:
		return
	var yv := d.normalized()
	var ref := Vector3.UP if abs(yv.y) < 0.9 else Vector3.RIGHT
	var xv := yv.cross(ref).normalized()
	var zv := xv.cross(yv)
	var cm := CylinderMesh.new()
	cm.top_radius = r
	cm.bottom_radius = r
	cm.height = d.length()
	cm.radial_segments = 8
	cm.rings = 1
	var mi := MeshInstance3D.new()
	mi.mesh = cm
	mi.material_override = m
	mi.transform = Transform3D(Basis(xv, yv, zv), (a + b) * 0.5)
	root.add_child(mi)


func _label(parent: Node3D, text: String, pos: Vector3, color := Color(1, 1, 1)) -> void:
	var l := Label3D.new()
	l.text = text
	l.font_size = 32
	l.pixel_size = 0.0007
	l.outline_size = 8
	l.modulate = color
	l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	l.no_depth_test = true
	l.position = pos
	parent.add_child(l)


func _sphere(parent: Node3D, p: Vector3, r: float, m: Material) -> void:
	var mi := MeshInstance3D.new()
	var s := SphereMesh.new()
	s.radius = r
	s.height = 2 * r
	s.radial_segments = 12
	s.rings = 6
	mi.mesh = s
	mi.material_override = m
	mi.position = p
	parent.add_child(mi)


func _build_anchor(root: Node3D, o: Dictionary) -> void:
	var p: Vector3 = o["p"]
	_sphere(root, p, 0.012, mat_anchor)
	for ax in [Vector3.RIGHT, Vector3.UP, Vector3.BACK]:
		var mi := MeshInstance3D.new()
		var b := BoxMesh.new()
		b.size = Vector3(0.004, 0.004, 0.004) + ax.abs() * 0.08
		mi.mesh = b
		mi.position = p + ax * 0.04
		var c := Color(abs(ax.x), abs(ax.y), abs(ax.z), 0.9)
		mi.material_override = _mat(c)
		root.add_child(mi)
	_label(root, "%s\n%.2f %.2f %.2f" % [o["name"], p.x, p.y, p.z], p + Vector3(0, 0.07, 0), Color(1, 0.6, 0.9))


func _box_transform(o: Dictionary) -> Transform3D:
	var ax: Array = o["axes"]
	return Transform3D(Basis(ax[0], ax[1], ax[2]), o["center"])


func _build_box(root: Node3D, o: Dictionary, is_preview: bool) -> void:
	var t := _box_transform(o)
	var s: Vector3 = o["size"]
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = s
	mi.mesh = bm
	mi.transform = t
	mi.material_override = mat_preview if is_preview else mat_box
	root.add_child(mi)
	# edges
	var im := ImmediateMesh.new()
	im.surface_begin(Mesh.PRIMITIVE_LINES, mat_edge)
	var h := s * 0.5
	var c := []
	for i in range(8):
		c.append(t * Vector3(h.x * (1 if i & 1 else -1), h.y * (1 if i & 2 else -1), h.z * (1 if i & 4 else -1)))
	for e in [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]:
		im.surface_add_vertex(c[e[0]])
		im.surface_add_vertex(c[e[1]])
	im.surface_end()
	var lines := MeshInstance3D.new()
	lines.mesh = im
	root.add_child(lines)
	var nm: String = o.get("name", "")
	_label(root, "%s %.0f x %.0f x %.0f cm" % [nm, s.x * 100, s.y * 100, s.z * 100],
		t.origin + Vector3(0, s.length() * 0.5 + 0.05, 0), Color(0.6, 0.9, 1.0))


func _build_path(root: Node3D, pts: Array, diameter: float, is_preview: bool) -> void:
	var r := maxf(diameter * 0.5, 0.004)
	var m := mat_preview if is_preview else mat_path
	for i in range(pts.size()):
		_sphere(root, pts[i], r * 1.2, m)
		if i == 0:
			continue
		var a: Vector3 = pts[i - 1]
		var b: Vector3 = pts[i]
		var d := b - a
		if d.length() < 1e-4:
			continue
		var yv := d.normalized()
		var ref := Vector3.UP if abs(yv.y) < 0.9 else Vector3.RIGHT
		var xv := yv.cross(ref).normalized()
		var zv := xv.cross(yv)
		var cm := CylinderMesh.new()
		cm.top_radius = r
		cm.bottom_radius = r
		cm.height = d.length()
		cm.radial_segments = 10
		cm.rings = 1
		var mi := MeshInstance3D.new()
		mi.mesh = cm
		mi.material_override = m
		mi.transform = Transform3D(Basis(xv, yv, zv), (a + b) * 0.5)
		root.add_child(mi)
	if pts.size() >= 2:
		_label(root, "%.2f m" % path_length(pts), pts[pts.size() - 1] + Vector3(0, 0.06, 0), Color(1, 0.75, 0.4))


## In-progress object: clicked points + the current reticle position as the next point.
func set_preview(kind: String, pts: Array, cursor: Variant) -> void:
	for c in preview.get_children():
		c.queue_free()
	var all := pts.duplicate()
	if cursor != null:
		all.append(cursor)
	if all.is_empty():
		return
	for p in pts:
		_sphere(preview, p, 0.01, mat_anchor)
	if kind == "box":
		if all.size() == 2:
			_build_path(preview, all, 0.006, true)
		elif all.size() >= 3:
			var b := box_from_points(all)
			if not b.is_empty():
				_build_box(preview, b, true)
	elif kind == "path" and all.size() >= 2:
		_build_path(preview, all, 0.02, true)
	elif kind == "measure" and all.size() >= 2:
		_build_measure(preview, all[0], all[1], true)
	elif kind == "outline" and all.size() >= 2:
		var segs := []
		for i in range(all.size()):
			segs.append_array([all[i], all[(i + 1) % all.size()]])
		_lines(preview, segs, _mat(Color(1.0, 1.0, 0.3, 0.9)))


# ---------------------------------------------------------------- JSON

static func _v(v: Vector3) -> Array:
	return [snappedf(v.x, 1e-5), snappedf(v.y, 1e-5), snappedf(v.z, 1e-5)]


static func _vec(a: Variant) -> Vector3:
	return Vector3(a[0], a[1], a[2])


func to_dict() -> Dictionary:
	var out := []
	for o in objects:
		var d := {}
		for k in o:
			var v = o[k]
			if v is Vector3:
				d[k] = _v(v)
			elif v is Array:
				var arr := []
				for e in v:
					arr.append(_v(e) if e is Vector3 else e)
				d[k] = arr
			else:
				d[k] = v
		out.append(d)
	return {"format": FORMAT, "frame": "OpenXR STAGE == OpenVR Standing; metres; x right, y up, -z forward",
		"objects": out}


func from_dict(d: Dictionary) -> void:
	clear_all()
	for o in d.get("objects", []):
		var t: String = o.get("type", "")
		var n := {}
		for k in o:
			var v = o[k]
			if k in ["p", "center", "size", "c"]:
				n[k] = _vec(v)
			elif k in ["axes", "points", "centers"]:
				var arr := []
				for e in v:
					arr.append(_vec(e))
				n[k] = arr
			else:
				n[k] = v
		if t in counters:
			counters[t] += 1
			_add(n)
