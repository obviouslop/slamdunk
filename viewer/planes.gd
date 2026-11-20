class_name PlaneLayer
extends Node3D
## Coarse planes fitted to the fused cloud by tools/planes.py (/dev/shm/pcplanes.json): one 10 % translucent quad per plane
## (floor green, ceiling blue, wall orange, table yellow, other grey), double sided, no depth write so they never hide the cloud.
## Items are in STAGE coordinates (child of the XROrigin). Ids are stable on the server, so nodes are reused and only move.

const PATH := "/dev/shm/pcplanes.json"
const ALPHA := 0.10
const COLORS := {"floor": Color(0.2, 1.0, 0.3), "ceiling": Color(0.3, 0.5, 1.0), "wall": Color(1.0, 0.6, 0.15),
	"table": Color(1.0, 0.9, 0.2), "plane": Color(0.8, 0.8, 0.8)}
var path := PATH
var seq := -1
var nodes := {}              # id -> MeshInstance3D


func poll() -> bool:
	if not FileAccess.file_exists(path):
		return false
	var d = JSON.parse_string(FileAccess.get_file_as_string(path))
	if typeof(d) != TYPE_DICTIONARY or int(d.get("seq", -1)) == seq:
		return false
	seq = int(d["seq"])
	apply(d)
	return true


func apply(d: Dictionary) -> void:
	var seen := {}
	for it in d.get("items", []):
		var id := int(it["id"])
		seen[id] = true
		var mi: MeshInstance3D = nodes.get(id)
		var label := String(it.get("label", "plane"))
		if mi == null:
			mi = MeshInstance3D.new()
			mi.mesh = QuadMesh.new()
			mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			add_child(mi)
			nodes[id] = mi
		if String(mi.get_meta("label", "")) != label:
			mi.set_meta("label", label)
			mi.material_override = _material(COLORS.get(label, COLORS["plane"]))
		(mi.mesh as QuadMesh).size = Vector2(2.0 * float(it["hu"]), 2.0 * float(it["hv"]))
		var u := Vector3(it["u"][0], it["u"][1], it["u"][2])
		var v := Vector3(it["v"][0], it["v"][1], it["v"][2])
		var n := Vector3(it["n"][0], it["n"][1], it["n"][2])
		mi.transform = Transform3D(Basis(u, v, n), Vector3(it["centre"][0], it["centre"][1], it["centre"][2]))
	for k in nodes.keys():
		if not seen.has(k):
			nodes[k].queue_free()
			nodes.erase(k)


func _material(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = Color(c.r, c.g, c.b, ALPHA)
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
	m.render_priority = -1
	return m
