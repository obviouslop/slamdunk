extends SceneTree
## Headless checks (no XR, no window):
##   PCVIEW_SHM=<scratch> PCVIEW_SESSIONS_DIR=<scratch> godot --headless --path viewer --xr-mode off -s res://tests/test_headless.gd
##   (PCVIEW_SHM keeps the test away from a running session's /dev/shm files)
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
	var me := m2.add_measure(Vector3(0, 0, 0), Vector3(3, 4, 0))
	check(me["name"] == "M1" and "5.000 m" in SceneModel.measure_text(me["points"][0], me["points"][1]), "measure 5 m")
	var m3 := SceneModel.new()
	root.add_child(m3)
	m3.from_dict(JSON.parse_string(JSON.stringify(m2.to_dict())))
	check(m3.objects.size() == 3 and m3.objects[2]["type"] == "measure", "measure round trip")

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



var frames := 0

var inst: Node


func _process(_delta: float) -> bool:
	frames += 1
	if frames == 2:
		_late_a()
	elif frames == 4:
		_late_b()
		print("FAILS: %d" % fails)
	return frames > 30


func _late_a() -> void:
	# palette: hit test, tools, confirm for danger buttons
	var pal := Palette.new()
	root.add_child(pal)
	pal.add_button("a", "A", 0, 0, "tool")
	pal.add_button("x", "X", 1, 0, "danger")
	var got := []
	pal.pressed.connect(func(id): got.append(id))
	var ca: Vector2 = pal.buttons["a"]["rect"].get_center()
	var cx: Vector2 = pal.buttons["x"]["rect"].get_center()
	var h = pal.ray_hit(Vector3(ca.x, ca.y, 0.5), Vector3(0, 0, -1))
	check(h != null and h["id"] == "a" and abs(h["dist"] - 0.5) < 1e-5, "palette ray hit")
	check(pal.press() == "a" and got == ["a"], "palette press")
	pal.ray_hit(Vector3(cx.x, cx.y, 0.5), Vector3(0, 0, -1))
	check(pal.press() == "" and got.size() == 1, "danger needs confirmation")
	check(pal.press() == "x" and got == ["a", "x"], "danger confirmed on 2nd press")
	check(pal.ray_hit(Vector3(5, 5, 0.5), Vector3(0, 0, -1)) == null, "palette miss")

	# main scene script compiles and runs without XR
	OS.set_environment("PCVIEW_NO_XR", "1")
	OS.set_environment("PCVIEW_SESSIONS_DIR", OS.get_cache_dir().path_join("pcview_test_sessions"))   # never touch real data
	var scn: PackedScene = load("res://pc_main.tscn")
	inst = scn.instantiate()
	root.add_child(inst)


func _late_b() -> void:
	check(inst.get("tool") == "scan", "pc_main instantiated")
	var ip: Palette = inst.get("palette")
	check(ip.buttons.size() == 29, "toolbox has %d buttons" % ip.buttons.size())
	inst.call("_on_palette", "measure")
	check(inst.get("tool") == "measure", "toolbox selects tool")
	var nobj: int = inst.get("model").objects.size()
	inst.get("model").add_anchor(Vector3(0, 1, 0))
	ip.hovered = "clear_objects"
	ip.press()
	check(inst.get("model").objects.size() == nobj + 1, "clear objects not on first press")
	ip.hovered = "clear_objects"
	ip.press()
	check(inst.get("model").objects.size() == 0, "clear objects after confirmation")
	var seq: int = inst.get("cmd_seq")
	inst.call("_on_button", "x_button", "right_hand")
	var ctl = JSON.parse_string(FileAccess.get_file_as_string(inst.get("CTL_PATH")))
	check(ctl["cmd"] == "still" and int(ctl["cmd_seq"]) == seq + 1, "X sends still command")
	ip.hovered = "paint"
	inst.set("pal_hit", {"id": "paint"})
	inst.call("_on_button", "trigger_click", "right_hand")
	check(inst.get("tool") == "paint", "trigger click presses the hovered toolbox button")
	inst.call("_on_button_released", "trigger_click", "right_hand")
	check(inst.get("analog").get("right_hand/trigger") == false, "trigger click release")
	ip.hovered = "anchor"
	inst.set("pal_hit", {"id": "anchor"})
	inst.call("_on_button", "bumper", "right_hand")
	check(inst.get("tool") == "anchor", "bumper presses the hovered toolbox button")
	inst.set("pal_hit", null)
	inst.set("scanning", false)
	inst.call("_analog_pressed", "right_hand", "trigger")
	check(inst.call("_scan_state") == "paint", "analog trigger captures while scanning is stopped")
	inst.call("_analog_released", "right_hand", "trigger")
	check(inst.call("_scan_state") == "off", "released -> off")
	inst.set("scanning", true)
	var mfs := Transform3D(Basis(Vector3.UP, 0.1), Vector3(0.05, -0.02, 0.1))
	inst.set("stage_from_map_shown", mfs.affine_inverse())
	var pm: Vector3 = inst.call("_to_map", Vector3(1, 1, 1))
	check(pm.is_equal_approx(mfs * Vector3(1, 1, 1)), "stage -> map conversion for placed objects")
	inst.set("stage_from_map_shown", Transform3D.IDENTITY)
	inst.call("_on_button", "dpad_up", "left_hand")
	check(inst.get("scanning") == false, "D-pad up stops scanning")
	inst.call("_on_button", "dpad_up", "left_hand")
	# depth passthrough: frame written by tools/ptpub.py (fixture), CPU unprojection == numpy reference
	var dp: DepthPassthrough = inst.get("passthrough")
	dp.path = "res://tests/fixtures/pcpass_test.bin"
	inst.call("_on_palette", "pass")
	check(dp.visible and ip.buttons["pass"]["active"], "DEPTH VIEW toggles the passthrough")
	check(dp.poll() and dp.seq == 7 and dp.frame == 4242 and dp.grid == Vector2i(4, 4), "passthrough frame read")
	check(not dp.poll(), "same seq is not re-read")
	var cp = dp.cell_point(2, 3)
	check(cp != null and (cp as Vector3).is_equal_approx(Vector3(1.6, 1.0, -2.2)), "passthrough cell unprojection %s" % [cp])
	check(dp.cell_point(0, 0) == null, "cell without depth is invalid")
	check(dp.mesh != null and dp.mesh.get_surface_count() == 1, "passthrough grid mesh built")
	inst.call("_on_palette", "pass")
	check(not dp.visible, "DEPTH VIEW off")
	# SELECT: per-tool buttons, brush selection -> exclusion zone, undo; zone file for the server
	inst.call("_on_palette", "zone")
	check(ip.buttons["zone_mode"]["enabled"] and not ip.buttons["sel_delete"]["enabled"], "ZONE shows its own toolbox row")
	inst.call("_on_palette", "select")
	check(ip.buttons["sel_delete"]["enabled"] and not ip.buttons["zone_mode"]["enabled"], "SELECT shows its own toolbox row")
	var nz: int = inst.get("model").objects.size()
	inst.set("sel_centers", [Vector3(0, 1, -1), Vector3(0.05, 1, -1)])
	inst.set("sel_radii", [0.1, 0.1])
	inst.set("sel_strokes", [0])
	inst.call("_on_palette", "sel_delete")
	var mo: Array = inst.get("model").objects
	check(mo.size() == nz + 1 and mo[mo.size() - 1]["type"] == "zone" and mo[mo.size() - 1]["mode"] == "exclude", "DELETE makes an exclusion zone")
	check((inst.get("sel_centers") as Array).is_empty(), "selection cleared after DELETE")
	var zf: Dictionary = inst.call("_read_json", inst.get("ZONES_PATH"))
	check((zf.get("zones", []) as Array).size() >= 1 and zf["zones"][-1]["shape"] == "spheres", "zones written for the server")
	inst.call("_undo")
	check(inst.get("model").objects.size() == nz, "A undoes the zone")
	# ZONE: cut + outline + label mode
	inst.call("_on_palette", "zone")
	inst.call("_on_palette", "zone_mode")
	check(inst.get("zone_mode") == "include", "zone mode cycles to INCLUDE")
	inst.call("_on_palette", "zone_mode")
	check(inst.get("zone_mode") == "label" and ip.buttons["next_label"]["enabled"], "LABEL mode shows NEXT LABEL")
	inst.set("pts", [Vector3(0, 0, 0), Vector3(2, 0, 0), Vector3(2, 0, -2)])
	inst.set("zone_shape", "outline")
	inst.call("_finish")
	mo = inst.get("model").objects
	check(mo[mo.size() - 1]["shape"] == "prism" and mo[mo.size() - 1]["mode"] == "label" and mo[mo.size() - 1]["label"] == "wall", "outline -> label prism zone")
	inst.call("_on_palette", "zone_show")
	check(not inst.get("model").nodes[mo.size() - 1].visible, "ZONES OFF hides zones")
	inst.call("_on_palette", "zone_show")
	inst.call("_undo")
	inst.call("_on_palette", "scan")

