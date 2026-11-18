class_name CloudTiles
extends Node3D
## Tiled point cloud from tools/pcpub.py (/dev/shm/pointcloud.bin, format PCLD0002).
## One MultiMeshInstance3D per tile (1 m cube by default) with a tight AABB (frustum culling);
## only tiles whose version changed are re-uploaded, at most `budget_points` per frame, so large
## clouds never stall the XR frame loop.

const MAGIC := "PCLD0002"
const HDR := 32
const ENTRY := 32

@export var path := "/dev/shm/pointcloud.bin"
@export var poll_s := 0.25
@export var budget_points := 40000

var material: Material
var quad: QuadMesh
var tiles := {}            # key -> {"node": MultiMeshInstance3D, "version": int, "count": int}
var last_seq := -1
var tile_size := 1.0
var total_points := 0
var last_update_ms := 0
var pending: Array = []    # [{key, version, offset, count}] from the file `pending_file`
var pending_file: FileAccess
var poll_accum := 0.0


func _init(mat: Material) -> void:
	material = mat
	quad = QuadMesh.new()
	quad.size = Vector2(1, 1)


func _process(delta: float) -> void:
	poll_accum += delta
	if poll_accum >= poll_s:
		poll_accum = 0.0
		_poll()
	_upload_some()


func clear() -> void:
	for k in tiles:
		tiles[k]["node"].queue_free()
	tiles.clear()
	pending.clear()
	total_points = 0


static func decode_key(key: int) -> Vector3i:
	return Vector3i((key & 0x1FFFFF) - (1 << 20), ((key >> 21) & 0x1FFFFF) - (1 << 20), ((key >> 42) & 0x1FFFFF) - (1 << 20))


func _poll() -> void:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return
	var hdr := f.get_buffer(HDR)
	if hdr.size() < HDR or hdr.slice(0, 8).get_string_from_ascii() != MAGIC:
		return
	var seq := hdr.decode_u64(16)
	if seq == last_seq:
		return
	var ntiles := hdr.decode_u32(8)
	tile_size = hdr.decode_float(24)
	var table := f.get_buffer(ntiles * ENTRY)
	if table.size() != ntiles * ENTRY:
		return
	last_seq = seq
	total_points = hdr.decode_u32(12)
	last_update_ms = Time.get_ticks_msec()
	var seen := {}
	pending.clear()
	for i in range(ntiles):
		var o := i * ENTRY
		var key := table.decode_s64(o)
		var version := table.decode_u64(o + 8)
		seen[key] = true
		if tiles.has(key) and tiles[key]["version"] == version:
			continue
		pending.append({"key": key, "version": version, "offset": table.decode_u64(o + 16), "count": table.decode_u32(o + 24)})
	pending_file = f       # keep the handle: the inode stays valid after the server renames a new file in
	for k in tiles.keys():
		if not seen.has(k):
			tiles[k]["node"].queue_free()
			tiles.erase(k)


func _upload_some() -> void:
	var done := 0
	while not pending.is_empty() and done < budget_points:
		var e: Dictionary = pending.pop_front()
		var count: int = e["count"]
		pending_file.seek(e["offset"])
		var body := pending_file.get_buffer(count * 64)
		if body.size() != count * 64:
			pending.clear()
			last_seq = -1          # file inconsistent: re-read next poll
			return
		var key: int = e["key"]
		var t: Dictionary = tiles.get(key, {})
		if t.is_empty():
			var inst := MultiMeshInstance3D.new()
			var mm := MultiMesh.new()
			mm.transform_format = MultiMesh.TRANSFORM_3D
			mm.use_colors = true
			mm.mesh = quad
			inst.multimesh = mm
			inst.material_override = material
			inst.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			var c := decode_key(key)
			inst.custom_aabb = AABB(Vector3(c) * tile_size - Vector3.ONE * 0.05, Vector3.ONE * (tile_size + 0.1))
			add_child(inst)
			t = {"node": inst}
			tiles[key] = t
		var mmi: MultiMeshInstance3D = t["node"]
		mmi.multimesh.instance_count = count
		mmi.multimesh.buffer = body.to_float32_array()
		t["version"] = e["version"]
		t["count"] = count
		done += count
