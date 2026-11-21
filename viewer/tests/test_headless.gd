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
	check(ip.buttons.size() == 37, "toolbox has %d buttons" % ip.buttons.size())
	check(not bool(inst.get("scanning")), "point-cloud scanning starts OFF")
	check(not bool(inst.get("rec_on")) and ip.buttons.has("liverec"), "REC button present, recording off")
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
	seq = inst.get("cmd_seq")
	inst.call("_on_button", "bumper", "left_hand")
	ctl = JSON.parse_string(FileAccess.get_file_as_string(inst.get("CTL_PATH")))
	check(ctl["cmd"] == "still" and int(ctl["cmd_seq"]) == seq + 1, "left bumper sends still command (dedicated STILL button)")
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
	var cam := Transform3D(dp.mat.get_shader_parameter("stage_from_rect")).origin
	var pp: Vector3 = cp
	check(dp.occludes(cam + (pp - cam) * 1.3) and not dp.occludes(cam + (pp - cam) * 0.7), "occlusion: points behind the measured surface fade, in front stay")
	var om := ShaderMaterial.new()
	om.shader = load("res://pc_points.gdshader")
	dp.apply_occlusion(om)
	check(om.get_shader_parameter("occ_depth") == dp.depth_tex and float(om.get_shader_parameter("occ_block")) == float(dp.mat.get_shader_parameter("block")), "occlusion uniforms handed to the cloud shader")
	inst.call("_on_palette", "pass")
	check(not dp.visible, "DEPTH VIEW off")
	# tag labels (fiducial axes + ID, QR text) from the server's pctags.json
	var tl: TagLayer = inst.get("tag_layer")
	tl.apply({"seq": 1, "items": [
		{"key": "fid:3:0", "kind": "fid", "id": 3, "T": [1, 0, 0, 0.5, 0, 1, 0, 1.0, 0, 0, 1, -2.0], "size": 0.05, "ntag": 4},
		{"key": "qr:hi:1", "kind": "qr", "text": "Hello", "T": [0, 0, 1, 1.0, 1, 0, 0, 1.5, 0, 1, 0, -1.0], "size": 0.1}]})
	check(tl.nodes.size() == 2 and (tl.nodes["fid:3:0"] as Node3D).position.is_equal_approx(Vector3(0.5, 1.0, -2.0)), "tag labels created at their STAGE poses")
	var qrl := ""
	for c in tl.nodes["qr:hi:1"].get_children():
		if c is Label3D:
			qrl = (c as Label3D).text
	check(qrl == "Hello", "QR text label")
	tl.apply({"seq": 2, "items": []})
	await inst.get_tree().process_frame
	check(tl.nodes.is_empty(), "tag labels removed when the server drops them")
	inst.call("_on_palette", "tags")
	check(not tl.visible and not ip.buttons["tags"]["active"], "TAGS toggle hides the labels")
	check(ip.buttons["far"]["active"] and bool(inst.get("far_on")), "FAR is on by default")
	inst.call("_on_palette", "far")
	check(not bool(inst.get("far_on")) and not ip.buttons["far"]["active"], "FAR toggle switches the far-range depth off")
	inst.call("_on_palette", "far")
	inst.call("_on_palette", "tags")
	# coarse planes: 10 % translucent quads from the server's pcplanes.json
	var pl: PlaneLayer = inst.get("plane_layer")
	check(not pl.visible and not bool(inst.get("planes_on")), "PLANES off by default")
	pl.apply({"seq": 1, "items": [
		{"id": 1, "label": "floor", "n": [0, 1, 0], "d": 0.0, "centre": [0.5, 0.0, -1.0], "u": [0, 0, -1], "v": [-1, 0, 0], "hu": 2.0, "hv": 1.5},
		{"id": 2, "label": "wall", "n": [0, 0, 1], "d": 2.0, "centre": [0.0, 1.2, -2.0], "u": [1, 0, 0], "v": [0, 1, 0], "hu": 2.0, "hv": 1.25}]})
	var fl: MeshInstance3D = pl.nodes[1]
	check(pl.nodes.size() == 2 and fl.global_transform.origin.is_equal_approx(Vector3(0.5, 0, -1)), "plane quads placed at their STAGE poses")
	check(is_equal_approx((fl.material_override as StandardMaterial3D).albedo_color.a, 0.10), "planes are 10 % opaque")
	check(fl.global_transform.basis.z.is_equal_approx(Vector3(0, 1, 0)) and (fl.mesh as QuadMesh).size.is_equal_approx(Vector2(4.0, 3.0)), "quad normal / extents follow the plane")
	pl.apply({"seq": 2, "items": [pl_item_keep()]})
	await inst.get_tree().process_frame
	check(pl.nodes.size() == 1, "vanished planes are removed")
	inst.call("_on_palette", "planes")
	check(pl.visible and ip.buttons["planes"]["active"], "PLANES toggle shows the planes")
	inst.call("_on_palette", "planes")
	# splat capture: independent of the point-cloud scanning toggle (neither / points / splat / both)
	check(not bool(inst.get("splat_on")) and bool(inst.get("scanning")), "defaults: POINTS on, SPLAT off")
	inst.call("_on_palette", "splat")
	check(bool(inst.get("splat_on")) and bool(inst.get("scanning")) and ip.buttons["splat"]["active"], "SPLAT toggles on without touching POINTS")
	check(bool(inst.call("_read_json", inst.get("CTL_PATH")).get("splat", false)), "splat flag written for the server")
	inst.call("_on_palette", "rec")
	check(not bool(inst.get("scanning")) and bool(inst.get("splat_on")), "POINTS off leaves SPLAT on")
	inst.call("_on_palette", "rec")
	inst.call("_on_palette", "splat")
	check(not bool(inst.get("splat_on")), "SPLAT off again")
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



func pl_item_keep() -> Dictionary:
	return {"id": 2, "label": "wall", "n": [0, 0, 1], "d": 2.0, "centre": [0.0, 1.2, -2.0], "u": [1, 0, 0], "v": [0, 1, 0], "hu": 2.0, "hv": 1.25}
