extends Node3D
## Point cloud + planning viewer for the Steam Frame: OpenXR (SteamVR), Vulkan, alpha-blend passthrough.
##
## Cloud: tiles published by tools/pcserver.py at /dev/shm/pointcloud.bin (CloudTiles). Coordinates are
## OpenVR Standing == OpenXR STAGE, so XROrigin3D stays at identity.
## Planning objects (SceneModel): anchors, boxes, paths; placed on the scanned surface under the right
## controller ray (server raycast, /dev/shm/pchit.json) or freely at the ray tip.
##
## Right controller   A        next tool: SCAN > PAINT > ANCHOR > BOX > PATH
##                    trigger  PAINT: hold to capture where you point; ANCHOR/BOX/PATH: place a point
##                    B        SCAN/PAINT: point size; ANCHOR/BOX/PATH: undo (last point, else last object)
##                    grip     BOX: finish as panel after 3 points; PATH: finish the path
##                    stick    up/down: free-placement distance; click: surface snap on/off
## Left controller    trigger  transparency      X  show/hide cloud     Y  save (cloud .ply + scene .json)
##                    stick click: scanning on/off   grip hold 1 s: clear cloud, 3 s: also clear objects
##
## Files: /dev/shm/pcctl.json (to server), /dev/shm/pchit.json, /dev/shm/pcstatus.json (from server),
## /dev/shm/gnss.json + /dev/shm/georef.json (tools/gnssd.py, optional),
## captures/sessions/autosave_scene.json (every change; reloaded at start), captures/sessions/<name>.json

const CTL_PATH := "/dev/shm/pcctl.json"
const HIT_PATH := "/dev/shm/pchit.json"
const STATUS_PATH := "/dev/shm/pcstatus.json"
const GNSS_PATH := "/dev/shm/gnss.json"
const GEOREF_PATH := "/dev/shm/georef.json"
const CTL_S := 1.0 / 30.0
const TOOLS := ["scan", "paint", "anchor", "box", "path"]

var xr_interface: OpenXRInterface
var origin: XROrigin3D
var camera: XRCamera3D
var cloud: CloudTiles
var model: SceneModel
var mat: ShaderMaterial
var status: Label3D
var reticle: MeshInstance3D
var ret_mat: StandardMaterial3D
var ctrl := {}
var rays := {}
var sizes := [0.004, 0.006, 0.010, 0.016]
var size_idx := 1
var alphas := [0.3, 0.55, 0.8, 1.0]
var alpha_idx := 1

var tool := "scan"
var scanning := true
var painting := false
var snap := true
var free_dist := 0.6
var pts: Array = []          # clicked points of the object in progress
var cmd := ""
var cmd_seq := 0
var save_name := ""
var ctl_accum := 0.0
var hit := {}                # latest pchit.json
var hit_seq := -1
var hit_rx_ms := 0
var srv := {}
var gnss := {}
var georef := {}
var slow_accum := 0.0
var msg := ""
var msg_ms := 0
var grip_down_ms := -1
var grip_stage := 0
var sessions_dir := ""


func _ready() -> void:
	sessions_dir = ProjectSettings.globalize_path("res://").path_join("../captures/sessions").simplify_path()
	DirAccess.make_dir_recursive_absolute(sessions_dir)
	_build_scene()
	_load_autosave()
	xr_interface = XRServer.find_interface("OpenXR")
	if xr_interface and xr_interface.is_initialized():
		get_viewport().use_xr = true
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		xr_interface.session_begun.connect(_on_session_begun)
	elif OS.has_environment("PCVIEW_NO_XR"):
		print("PCVIEW running without XR (test mode)")
	else:
		push_error("OpenXR not initialized")
		get_tree().quit()


func _build_scene() -> void:
	var env := WorldEnvironment.new()
	env.environment = Environment.new()
	env.environment.background_mode = Environment.BG_COLOR
	env.environment.background_color = Color(0, 0, 0, 0)
	add_child(env)

	origin = XROrigin3D.new()
	add_child(origin)
	camera = XRCamera3D.new()
	origin.add_child(camera)
	for hand in ["left_hand", "right_hand"]:
		var c := XRController3D.new()
		c.tracker = hand
		c.pose = &"aim"
		c.button_pressed.connect(_on_button.bind(hand))
		c.button_released.connect(_on_release.bind(hand))
		ctrl[hand] = c
		var ray := MeshInstance3D.new()
		var m := BoxMesh.new()
		m.size = Vector3(0.003, 0.003, 1.0)
		ray.mesh = m
		var rm := StandardMaterial3D.new()
		rm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		rm.albedo_color = Color(0.3, 0.9, 1.0)
		ray.material_override = rm
		c.add_child(ray)
		rays[hand] = ray
		origin.add_child(c)
	_set_ray("left_hand", 0.15)

	mat = ShaderMaterial.new()
	mat.shader = load("res://pc_points.gdshader")
	mat.set_shader_parameter("point_size", sizes[size_idx])
	mat.set_shader_parameter("alpha", alphas[alpha_idx])
	cloud = CloudTiles.new(mat)
	origin.add_child(cloud)

	model = SceneModel.new()
	model.changed.connect(_autosave)
	origin.add_child(model)

	reticle = MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.008
	sm.height = 0.016
	reticle.mesh = sm
	ret_mat = SceneModel._mat(Color(0.3, 1.0, 0.4, 0.9))
	ret_mat.no_depth_test = true
	reticle.material_override = ret_mat
	reticle.visible = false
	origin.add_child(reticle)

	status = Label3D.new()
	status.font_size = 28
	status.pixel_size = 0.0006
	status.no_depth_test = true
	status.modulate = Color(0.6, 1.0, 0.7)
	status.outline_size = 6
	status.position = Vector3(0, -0.17, -0.55)
	camera.add_child(status)


func _on_session_begun() -> void:
	var modes = xr_interface.get_supported_environment_blend_modes()
	if XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND in modes:
		xr_interface.environment_blend_mode = XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND
		get_viewport().transparent_bg = true
	else:
		push_warning("alpha blend passthrough not supported: %s" % [modes])
	print("PCVIEW session begun, blend modes ", modes, " refresh ", xr_interface.get_display_refresh_rate())


# ------------------------------------------------------------------ input

func _say(s: String) -> void:
	msg = s
	msg_ms = Time.get_ticks_msec()
	print("PCVIEW ", s)


func _buzz(hand := "right_hand", amp := 0.4, dur := 0.04) -> void:
	var c: XRController3D = ctrl.get(hand)
	if c:
		c.trigger_haptic_pulse("haptic", 0.0, amp, dur, 0.0)


func _send_cmd(c: String) -> void:
	cmd = c
	cmd_seq += 1
	_write_ctl()


func _on_button(action: String, hand: String) -> void:
	if hand == "right_hand":
		match action:
			"ax_button":
				_set_tool(TOOLS[(TOOLS.find(tool) + 1) % TOOLS.size()])
			"trigger_click":
				if tool == "paint":
					painting = true
				elif tool in ["anchor", "box", "path"]:
					_place()
			"by_button":
				if tool in ["scan", "paint"]:
					size_idx = (size_idx + 1) % sizes.size()
					mat.set_shader_parameter("point_size", sizes[size_idx])
				else:
					_undo()
			"grip_click":
				_finish()
			"primary_click":
				snap = not snap
				_say("surface snap " + ("on" if snap else "off"))
	else:
		match action:
			"trigger_click":
				alpha_idx = (alpha_idx + 1) % alphas.size()
				mat.set_shader_parameter("alpha", alphas[alpha_idx])
			"ax_button":
				cloud.visible = not cloud.visible
			"by_button":
				_save()
			"primary_click":
				scanning = not scanning
				_say("scanning " + ("on" if scanning else "paused"))
			"grip_click":
				grip_down_ms = Time.get_ticks_msec()
				grip_stage = 0
	_write_ctl()


func _on_release(action: String, hand: String) -> void:
	if hand == "right_hand" and action == "trigger_click":
		painting = false
	if hand == "left_hand" and action == "grip_click":
		grip_down_ms = -1
	_write_ctl()


func _set_tool(t: String) -> void:
	if pts.size() > 0:
		_finish()
	tool = t
	painting = false
	pts.clear()
	model.set_preview("", [], null)
	_say("tool " + tool.to_upper())


func _cursor() -> Variant:
	## Placement point: scanned surface under the right ray (snap on, fresh hit), else the free ray tip.
	var c: XRController3D = ctrl.get("right_hand")
	if c == null or not c.get_has_tracking_data():
		return null
	var t := c.global_transform
	var dir := -t.basis.z
	if snap and hit.get("hit", false) and Time.get_ticks_msec() - hit_rx_ms < 500:
		return t.origin + dir * float(hit["dist"])
	return t.origin + dir * free_dist


func _place() -> void:
	var p = _cursor()
	if p == null:
		return
	_buzz()
	match tool:
		"anchor":
			var o := model.add_anchor(p)
			_say("anchor %s" % o["name"])
		"box":
			pts.append(p)
			if pts.size() == 4:
				var o := model.add_box(pts)
				_say("box %s" % o.get("name", "failed (degenerate)"))
				pts.clear()
		"path":
			pts.append(p)


func _finish() -> void:
	if tool == "box" and pts.size() == 3:
		var o := model.add_box(pts)
		_say("panel %s" % o.get("name", "failed (degenerate)"))
	elif tool == "path" and pts.size() >= 2:
		var o := model.add_path(pts)
		_say("path %s %.2f m" % [o["name"], SceneModel.path_length(pts)])
	pts.clear()
	model.set_preview("", [], null)


func _undo() -> void:
	if pts.size() > 0:
		pts.pop_back()
		_say("undo point")
	else:
		var o := model.remove_last()
		_say("removed %s" % o.get("name", "nothing"))
	_buzz("right_hand", 0.2, 0.02)


# ------------------------------------------------------------------ files

func _write_json(path: String, d: Dictionary) -> void:
	var tmp := path + ".tmp"
	var f := FileAccess.open(tmp, FileAccess.WRITE)
	if f == null:
		return
	f.store_string(JSON.stringify(d, " "))
	f.close()
	DirAccess.rename_absolute(tmp, path)


func _read_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	var v = JSON.parse_string(f.get_as_text())
	return v if v is Dictionary else {}


func _scan_state() -> String:
	if not scanning:
		return "off"
	if tool == "paint":
		return "paint"
	return "auto"


func _write_ctl() -> void:
	var aim := []
	var c: XRController3D = ctrl.get("right_hand")
	if c != null and c.get_has_tracking_data():
		var t := c.global_transform        # XROrigin at identity -> STAGE space
		for r in range(3):
			aim.append_array([t.basis.x[r], t.basis.y[r], t.basis.z[r], t.origin[r]])
	var d := {"t_ms": Time.get_ticks_msec(), "scan": _scan_state(), "painting": painting and aim.size() == 12,
		"aim": aim, "cmd": cmd, "cmd_seq": cmd_seq, "name": save_name}
	var f := FileAccess.open(CTL_PATH, FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(d))
		f.close()


func _scene_dict() -> Dictionary:
	var d := model.to_dict()
	d["created"] = Time.get_datetime_string_from_system()
	if not georef.is_empty():
		d["georef"] = georef
	return d


func _autosave() -> void:
	_write_json(sessions_dir.path_join("autosave_scene.json"), _scene_dict())


func _load_autosave() -> void:
	var d := _read_json(sessions_dir.path_join("autosave_scene.json"))
	if not d.is_empty():
		model.from_dict(d)
		print("PCVIEW loaded %d objects from autosave" % model.objects.size())


func _save() -> void:
	var dt := Time.get_datetime_dict_from_system()
	save_name = "scan_%04d%02d%02d_%02d%02d%02d" % [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
	var d := _scene_dict()
	d["cloud"] = save_name + ".ply"
	_write_json(sessions_dir.path_join(save_name + ".json"), d)
	_send_cmd("save")
	_buzz("left_hand", 0.6, 0.1)
	_say("saved " + save_name)


# ------------------------------------------------------------------ per frame

func _set_ray(hand: String, length: float, color := Color(0.3, 0.9, 1.0)) -> void:
	var r: MeshInstance3D = rays.get(hand)
	if r == null:
		return
	(r.mesh as BoxMesh).size = Vector3(0.003, 0.003, length)
	r.position = Vector3(0, 0, -length * 0.5)
	(r.material_override as StandardMaterial3D).albedo_color = color


func _process(delta: float) -> void:
	ctl_accum += delta
	if ctl_accum >= CTL_S:
		ctl_accum = 0.0
		_write_ctl()
		var h := _read_json(HIT_PATH)
		if not h.is_empty() and int(h.get("seq", -1)) != hit_seq:
			hit = h
			hit_seq = int(h["seq"])
			hit_rx_ms = Time.get_ticks_msec()
	slow_accum += delta
	if slow_accum >= 0.5:
		slow_accum = 0.0
		srv = _read_json(STATUS_PATH)
		gnss = _read_json(GNSS_PATH)
		georef = _read_json(GEOREF_PATH)

	# free-placement distance with the right stick
	var rc: XRController3D = ctrl.get("right_hand")
	if rc:
		var st := rc.get_vector2("primary")
		if abs(st.y) > 0.3:
			free_dist = clampf(free_dist * (1.0 + st.y * delta * 1.5), 0.15, 8.0)

	# left grip hold: 1 s clear cloud, 3 s also objects
	if grip_down_ms >= 0:
		var held := (Time.get_ticks_msec() - grip_down_ms) / 1000.0
		if held > 1.0 and grip_stage == 0:
			grip_stage = 1
			_send_cmd("reset")
			cloud.clear()
			_buzz("left_hand", 0.5, 0.08)
			_say("cloud cleared (keep holding 2 s to clear objects)")
		elif held > 3.0 and grip_stage == 1:
			grip_stage = 2
			model.clear_all()
			pts.clear()
			_buzz("left_hand", 1.0, 0.2)
			_say("objects cleared")

	# ray, reticle, preview
	var modeling: bool = tool in ["anchor", "box", "path"]
	var cur = _cursor() if modeling else null
	if rc and cur != null:
		var on_surface: bool = snap and hit.get("hit", false) and Time.get_ticks_msec() - hit_rx_ms < 500
		reticle.visible = true
		reticle.position = cur
		ret_mat.albedo_color = Color(0.3, 1.0, 0.4, 0.95) if on_surface else Color(1.0, 0.6, 0.1, 0.95)
		_set_ray("right_hand", (cur - rc.global_transform.origin).length(), Color(0.6, 1.0, 0.6) if on_surface else Color(1.0, 0.7, 0.3))
		if tool in ["box", "path"]:
			model.set_preview(tool, pts, cur)
	else:
		reticle.visible = false
		if tool == "paint":
			_set_ray("right_hand", 2.0 if painting else 0.3, Color(1.0, 0.25, 0.2) if painting else Color(1.0, 0.6, 0.1))
		else:
			_set_ray("right_hand", 0.3)

	_update_status()


func _hint() -> String:
	match tool:
		"scan":
			return "A: next tool  B: point size"
		"paint":
			return "hold trigger to capture"
		"anchor":
			return "trigger: place anchor  B: undo"
		"box":
			var steps := ["corner 1", "corner 2 (first edge)", "width (or grip: panel)", "height"]
			return "trigger: %s  B: undo" % steps[mini(pts.size(), 3)]
		"path":
			return "trigger: add point (%d)  grip: finish  B: undo" % pts.size()
	return ""


func _update_status() -> void:
	var lines := []
	var age := (Time.get_ticks_msec() - cloud.last_update_ms) / 1000.0 if cloud.last_update_ms > 0 else -1.0
	lines.append("%s%s   %s" % [tool.to_upper(), " (CAPTURING)" if painting else "", _hint()])
	lines.append("cloud %d pts  age %.1fs  %s %.1f/s   %d fps  objects %d%s" % [cloud.total_points, age,
		"scan" if scanning else "PAUSED", float(srv.get("fps", 0.0)), Engine.get_frames_per_second(),
		model.objects.size(), "" if cloud.visible else "  (cloud hidden)"])
	if not gnss.is_empty():
		var g := "GNSS %s  sats %d  hacc %.2f m" % [gnss.get("fix", "?"), int(gnss.get("sats", 0)), float(gnss.get("hacc", -1))]
		if georef.get("ok", false):
			g += "  georef rms %.2f m (n=%d)" % [float(georef.get("rms", 0)), int(georef.get("n", 0))]
		lines.append(g)
	if Time.get_ticks_msec() - msg_ms < 4000:
		lines.append(msg)
	status.text = "\n".join(lines)
