extends Node3D
## Point cloud viewer for the Steam Frame: OpenXR (SteamVR), Vulkan, alpha-blend passthrough.
## Shows the cloud published by tools/pcpub.py at /dev/shm/pointcloud.bin (MultiMesh buffer layout,
## coordinates = OpenVR Standing == OpenXR STAGE, so no transform is needed).
## Controls (right): trigger hold = paint-capture in paint mode, A = auto/paint mode, B = point size
## Controls (left):  trigger = transparency, X = show/hide cloud, Y = save PLY, grip = reset map
## Viewer -> server control: /dev/shm/pcctl.json (tools/pcserver.py), aim pose in STAGE (== OpenVR Standing).

const CLOUD_PATH := "/dev/shm/pointcloud.bin"
const POLL_S := 0.25
const CTL_PATH := "/dev/shm/pcctl.json"
const CTL_S := 0.1

var xr_interface: OpenXRInterface
var origin: XROrigin3D
var camera: XRCamera3D
var mm_inst: MultiMeshInstance3D
var mat: ShaderMaterial
var status: Label3D
var last_seq := -1
var last_count := 0
var last_update_ms := 0
var poll_accum := 0.0
var sizes := [0.004, 0.006, 0.010, 0.016]
var size_idx := 1
var alphas := [0.3, 0.55, 0.8, 1.0]
var alpha_idx := 1
var load_ms := 0
var ctrl := {}            # hand -> XRController3D
var rays := {}            # hand -> MeshInstance3D
var mode := "auto"        # auto | paint
var painting := false
var cmd := ""
var cmd_seq := 0
var ctl_accum := 0.0
var last_msg := ""


func _ready() -> void:
	_build_scene()
	xr_interface = XRServer.find_interface("OpenXR")
	if xr_interface and xr_interface.is_initialized():
		var vp := get_viewport()
		vp.use_xr = true
		DisplayServer.window_set_vsync_mode(DisplayServer.VSYNC_DISABLED)
		xr_interface.session_begun.connect(_on_session_begun)
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
		var m := BoxMesh.new(); m.size = Vector3(0.004, 0.004, 0.3)
		ray.mesh = m; ray.position = Vector3(0, 0, -0.15)
		var rm := StandardMaterial3D.new(); rm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED; rm.albedo_color = Color(0.3, 0.9, 1.0)
		ray.material_override = rm
		c.add_child(ray)
		rays[hand] = ray
		origin.add_child(c)

	mm_inst = MultiMeshInstance3D.new()
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	var quad := QuadMesh.new(); quad.size = Vector2(1, 1)
	mm.mesh = quad
	mm_inst.multimesh = mm
	mat = ShaderMaterial.new()
	mat.shader = load("res://pc_points.gdshader")
	mat.set_shader_parameter("point_size", sizes[size_idx])
	mat.set_shader_parameter("alpha", alphas[alpha_idx])
	mm_inst.material_override = mat
	mm_inst.custom_aabb = AABB(Vector3(-100, -100, -100), Vector3(200, 200, 200))
	origin.add_child(mm_inst)   # cloud lives in STAGE space = XROrigin space

	status = Label3D.new()
	status.font_size = 28
	status.pixel_size = 0.0006
	status.no_depth_test = true
	status.modulate = Color(0.6, 1.0, 0.7)
	status.outline_size = 6
	status.position = Vector3(0, -0.16, -0.55)
	camera.add_child(status)


func _on_session_begun() -> void:
	var modes = xr_interface.get_supported_environment_blend_modes()
	if XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND in modes:
		xr_interface.environment_blend_mode = XRInterface.XR_ENV_BLEND_MODE_ALPHA_BLEND
		get_viewport().transparent_bg = true
	else:
		push_warning("alpha blend passthrough not supported: %s" % [modes])
	var rate := xr_interface.get_display_refresh_rate()
	if rate > 0:
		Engine.physics_ticks_per_second = roundi(rate)
	print("PCVIEW session begun, blend modes ", modes, " refresh ", rate)


func _send_cmd(c: String) -> void:
	cmd = c
	cmd_seq += 1
	last_msg = c
	_write_ctl()


func _on_button(action: String, hand: String) -> void:
	if hand == "right_hand":
		if action == "trigger_click":
			painting = true
		elif action == "ax_button":
			mode = "paint" if mode == "auto" else "auto"
			last_msg = "mode " + mode
		elif action == "by_button":
			size_idx = (size_idx + 1) % sizes.size()
			mat.set_shader_parameter("point_size", sizes[size_idx])
	else:
		if action == "trigger_click":
			alpha_idx = (alpha_idx + 1) % alphas.size()
			mat.set_shader_parameter("alpha", alphas[alpha_idx])
		elif action == "ax_button":
			mm_inst.visible = not mm_inst.visible
		elif action == "by_button":
			_send_cmd("save")
		elif action == "grip_click":
			_send_cmd("reset")
	_update_rays()
	_write_ctl()


func _on_release(action: String, hand: String) -> void:
	if hand == "right_hand" and action == "trigger_click":
		painting = false
		_update_rays()
		_write_ctl()


func _update_rays() -> void:
	var r: MeshInstance3D = rays.get("right_hand")
	if r == null:
		return
	var m: StandardMaterial3D = r.material_override
	if mode == "paint" and painting:
		m.albedo_color = Color(1.0, 0.25, 0.2)
		(r.mesh as BoxMesh).size = Vector3(0.006, 0.006, 2.0); r.position = Vector3(0, 0, -1.0)
	else:
		m.albedo_color = Color(1.0, 0.6, 0.1) if mode == "paint" else Color(0.3, 0.9, 1.0)
		(r.mesh as BoxMesh).size = Vector3(0.004, 0.004, 0.3); r.position = Vector3(0, 0, -0.15)


func _write_ctl() -> void:
	var aim := []
	var c: XRController3D = ctrl.get("right_hand")
	if c != null and c.get_has_tracking_data():
		var t := c.global_transform        # XROrigin at identity -> STAGE space
		for r in range(3):
			aim.append_array([t.basis.x[r], t.basis.y[r], t.basis.z[r], t.origin[r]])
	var d := {"t_ms": Time.get_ticks_msec(), "mode": mode, "painting": painting and aim.size() == 12,
		"aim": aim, "cmd": cmd, "cmd_seq": cmd_seq}
	var f := FileAccess.open(CTL_PATH, FileAccess.WRITE)
	if f:
		f.store_string(JSON.stringify(d))
		f.close()


var cam_print_accum := 0.0

func _process(delta: float) -> void:
	cam_print_accum += delta
	if OS.has_environment("PCVIEW_DEBUG_CAM") and cam_print_accum > 1.0:
		cam_print_accum = 0.0
		var t := camera.global_transform
		print("PCVIEW camera t=%.3f pos %.3f %.3f %.3f fwd %.3f %.3f %.3f" % [Time.get_ticks_msec() / 1000.0, t.origin.x, t.origin.y, t.origin.z, -t.basis.z.x, -t.basis.z.y, -t.basis.z.z])
	ctl_accum += delta
	if ctl_accum >= CTL_S:
		ctl_accum = 0.0
		_write_ctl()
	poll_accum += delta
	if poll_accum >= POLL_S:
		poll_accum = 0.0
		_poll_cloud()
	var age := (Time.get_ticks_msec() - last_update_ms) / 1000.0 if last_update_ms > 0 else -1.0
	status.text = "%s%s  points %d  age %.1fs\n%d fps  size %.0f mm  alpha %.2f%s  %s" % [
		mode.to_upper(), " (PAINTING)" if painting and mode == "paint" else "", last_count, age,
		Engine.get_frames_per_second(), sizes[size_idx] * 1000, alphas[alpha_idx],
		"" if mm_inst.visible else "  (hidden)", last_msg]


func _poll_cloud() -> void:
	var f := FileAccess.open(CLOUD_PATH, FileAccess.READ)
	if f == null:
		return
	var hdr := f.get_buffer(32)
	if hdr.size() < 32 or hdr.slice(0, 8).get_string_from_ascii() != "PCLD0001":
		return
	var count := hdr.decode_u32(8)
	var seq := hdr.decode_u64(16)
	if seq == last_seq:
		return
	var t0 := Time.get_ticks_msec()
	var body := f.get_buffer(count * 64)
	if body.size() != count * 64:
		return
	var mm := mm_inst.multimesh
	mm.instance_count = count
	mm.buffer = body.to_float32_array()
	last_seq = seq
	last_count = count
	last_update_ms = Time.get_ticks_msec()
	load_ms = last_update_ms - t0
	print("PCVIEW loaded cloud seq %d: %d points in %d ms" % [seq, count, load_ms])
