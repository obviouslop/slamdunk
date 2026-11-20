class_name TagLayer
extends Node3D
## Labels for tags in view, published by tools/tags.py (/dev/shm/pctags.json):
##  * fiducials (AprilTag): their origin, tiny XYZ axes (x red, y green, z blue) and the ID
##  * QR codes: the decoded text, drawn flat on the code (dark backing, fitted to its size)
## Items are in STAGE coordinates; this node must be a child of the XROrigin (identity). The server keeps items alive for a while after
## they were last seen, so labels do not flicker; they are removed when the server drops them.

const PATH := "/dev/shm/pctags.json"
var path := PATH
var seq := -1
var nodes := {}              # key -> Node3D


func poll() -> bool:
	if not FileAccess.file_exists(path):
		return false
	var d = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(d) != TYPE_DICTIONARY or int(d.get("seq", -1)) == seq:
		return false
	seq = int(d["seq"])
	apply(d)
	return true


static func _xf(t: Array) -> Transform3D:
	return Transform3D(Basis(Vector3(t[0], t[4], t[8]), Vector3(t[1], t[5], t[9]), Vector3(t[2], t[6], t[10])), Vector3(t[3], t[7], t[11]))


func apply(d: Dictionary) -> void:
	var seen := {}
	for it in d.get("items", []):
		var key := String(it["key"])
		seen[key] = true
		var n: Node3D = nodes.get(key)
		var sig := "%s|%s|%.2f" % [it.get("kind", ""), it.get("text", it.get("id", "")), float(it.get("size", 0.0))]
		if n != null and String(n.get_meta("sig")) != sig:
			n.queue_free()
			n = null
		if n == null:
			n = _make_fid(it) if it["kind"] == "fid" else _make_qr(it)
			n.set_meta("sig", sig)
			add_child(n)
			nodes[key] = n
		n.transform = _xf(it["T"])
	for k in nodes.keys():
		if not seen.has(k):
			nodes[k].queue_free()
			nodes.erase(k)


func _unshaded(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.albedo_color = c
	m.render_priority = 5
	if c.a < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m


func _label(text: String, px: float, font: int, col: Color, billboard: bool) -> Label3D:
	var l := Label3D.new()
	l.text = text
	l.pixel_size = px
	l.font_size = font
	l.modulate = col
	l.outline_size = 8
	l.outline_modulate = Color(0, 0, 0, 0.9)
	l.render_priority = 6
	l.outline_render_priority = 5
	if billboard:
		l.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	return l


func _make_fid(it: Dictionary) -> Node3D:
	var root := Node3D.new()
	var L := clampf(0.5 * float(it.get("size", 0.05)), 0.02, 0.08)
	var axes := [[Vector3(1, 0, 0), Color(1.0, 0.25, 0.25), "x"], [Vector3(0, 1, 0), Color(0.3, 1.0, 0.3), "y"], [Vector3(0, 0, 1), Color(0.35, 0.55, 1.0), "z"]]
	for a in axes:
		var mi := MeshInstance3D.new()
		var bm := BoxMesh.new()
		var dir: Vector3 = a[0]
		bm.size = Vector3(L if dir.x > 0 else 0.0025, L if dir.y > 0 else 0.0025, L if dir.z > 0 else 0.0025)
		mi.mesh = bm
		mi.material_override = _unshaded(a[1])
		mi.position = dir * L * 0.5
		root.add_child(mi)
		var t := _label(a[2], 0.00035, 20, a[1], true)
		t.position = dir * (L + 0.006)
		root.add_child(t)
	var dot := MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.004
	sm.height = 0.008
	dot.mesh = sm
	dot.material_override = _unshaded(Color(1, 1, 1))
	root.add_child(dot)
	var id := _label("ID %d" % int(it["id"]), 0.0005, 32, Color(1.0, 0.95, 0.5), true)
	id.position = Vector3(L * 0.35, L * 0.35, 0.004) + Vector3(0, 0.012, 0)
	root.add_child(id)
	return root


func _make_qr(it: Dictionary) -> Node3D:
	var root := Node3D.new()
	var w := maxf(float(it.get("size", 0.08)), 0.03)
	var back := MeshInstance3D.new()
	var qm := QuadMesh.new()
	qm.size = Vector2(w, w)
	back.mesh = qm
	back.material_override = _unshaded(Color(0, 0, 0, 0.62))
	root.add_child(back)
	var px := w / 360.0
	var l := _label(String(it["text"]), px, 44, Color(1, 1, 1), false)
	l.width = 360.0
	l.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY
	l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	l.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	l.position = Vector3(0, 0, 0.002)
	root.add_child(l)
	return root
