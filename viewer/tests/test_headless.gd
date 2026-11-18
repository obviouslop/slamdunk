extends SceneTree
## Headless checks (no XR, no window):
##   godot --headless --path viewer --xr-mode off -s res://tests/test_headless.gd
## Needs a published cloud at /dev/shm/pointcloud.bin for the tile test (tools/pcpub.py some.ply).

var fails := 0

func check(cond: bool, what: String) -> void:
	if not cond:
		fails += 1
		print("FAIL ", what)
	else:
		print("ok   ", what)


func _initialize() -> void:
	# box geometry
	var b := SceneModel.box_from_points([Vector3(0, 0, 0), Vector3(1, 0.02, 0), Vector3(0.3, 0.01, -0.5), Vector3(0.5, 2.0, -0.2)])
	check(not b.is_empty(), "box built")
	check(b["size"].is_equal_approx(Vector3(1, 0.5, 2.0)), "upright box snapped size %s" % [b["size"]])
	check(b["center"].is_equal_approx(Vector3(0.5, 1.0, -0.25)), "upright box center %s" % [b["center"]])
	var p := SceneModel.box_from_points([Vector3(0, 1, 0), Vector3(1.7, 1, 0), Vector3(0, 1.6, -0.8)])
	check(abs(p["size"].x - 1.7) < 1e-4 and abs(p["size"].y - 1.0) < 1e-3 and abs(p["size"].z - 0.04) < 1e-6, "tilted panel %s" % [p["size"]])
	var ax: Array = p["axes"]
	check(abs(ax[0].dot(ax[1])) < 1e-5 and abs(ax[1].dot(ax[2])) < 1e-5 and abs(ax[0].cross(ax[1]).dot(ax[2]) - 1.0) < 1e-4, "panel axes orthonormal right-handed")

	# model + json round trip
	var m := SceneModel.new()
	root.add_child(m)
	m.add_anchor(Vector3(1, 2, 3))
	m.add_box([Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 0, -0.5), Vector3(0, 1, 0)])
	m.add_path([Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(1, 1, 0)])
	m.set_preview("box", [Vector3.ZERO, Vector3.RIGHT], Vector3(0, 0, -1))
	var d := m.to_dict()
	var txt := JSON.stringify(d)
	var m2 := SceneModel.new()
	root.add_child(m2)
	m2.from_dict(JSON.parse_string(txt))
	check(m2.objects.size() == 3, "json round trip count")
	check(m2.objects[0]["p"].is_equal_approx(Vector3(1, 2, 3)), "anchor round trip")
	check(m2.objects[1]["size"].is_equal_approx(m.objects[1]["size"]), "box round trip")
	check(abs(SceneModel.path_length(m2.objects[2]["points"]) - 2.0) < 1e-5, "path length")
	m2.remove_last()
	check(m2.objects.size() == 2, "undo")

	# tiles
	var mat := ShaderMaterial.new()
	mat.shader = load("res://pc_points.gdshader")
	var ct := CloudTiles.new(mat)
	root.add_child(ct)
	ct.budget_points = 1 << 30
	ct._poll()
	ct._upload_some()
	var n := 0
	for k in ct.tiles:
		n += ct.tiles[k]["count"]
	check(ct.tiles.size() > 0 and n == ct.total_points, "tiles loaded %d tiles, %d of %d points" % [ct.tiles.size(), n, ct.total_points])
	var t0 := Time.get_ticks_usec()
	ct.last_seq = -1
	ct._poll()
	check(ct.pending.is_empty(), "unchanged tiles not re-uploaded (poll %d us)" % (Time.get_ticks_usec() - t0))

	# main scene script compiles and runs without XR
	OS.set_environment("PCVIEW_NO_XR", "1")
	var scn: PackedScene = load("res://pc_main.tscn")
	var inst := scn.instantiate()
	root.add_child(inst)
	check(inst.get("tool") == "scan", "pc_main instantiated")
	print("FAILS: %d" % fails)


var frames := 0

func _process(_delta: float) -> bool:
	frames += 1
	return frames > 30
