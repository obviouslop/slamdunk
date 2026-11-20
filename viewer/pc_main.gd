extends Node3D
## Point cloud + planning viewer for the Steam Frame: OpenXR (SteamVR), Vulkan, alpha-blend passthrough.
##
## Cloud: tiles published by tools/pcserver.py at /dev/shm/pointcloud.bin (CloudTiles). Coordinates are
## OpenVR Standing == OpenXR STAGE, so XROrigin3D stays at identity.
## Planning objects (SceneModel): anchors, boxes/panels, paths, measurements; placed on the scanned surface
## under the right controller ray (server raycast, /dev/shm/pchit.json) or freely at the ray tip.
##
## Steam Frame controllers (native OpenXR profile /interaction_profiles/valve/frame_controller_valve):
##   RIGHT (tool hand)  trigger  use tool / press a toolbox button      A  undo        B  next tool
##                      grip or bumper  finish panel / path            X  stereo still  Y  help on/off
##                      stick up/down  free-placement distance          stick click  surface snap on/off
##   LEFT (toolbox)     toolbox panel floats above the controller (point at it with the right hand)
##                      D-pad up  scan REC / STOP      D-pad down  hide/show cloud    D-pad left/right  point size
##                      View  toolbox on/off           trigger (hold)  peek: hide the cloud while held
##                      stick up/down  cloud opacity
##   Destructive actions (clear cloud, clear objects, exit) only from the toolbox, with a confirming 2nd press.
##
## Files: /dev/shm/pcctl.json (to server), /dev/shm/pchit.json, /dev/shm/pcstatus.json (from server),
## /dev/shm/gnss.json + /dev/shm/georef.json (tools/gnssd.py, optional),
## captures/sessions/autosave_scene.json (every change; reloaded at start), captures/sessions/<name>.json,
## captures/stills/<time>/ (server, tools/stills.py)

# shared-memory files; PCVIEW_SHM redirects them (tests must never touch a running session's files)
var SHM: String = OS.get_environment("PCVIEW_SHM") if OS.has_environment("PCVIEW_SHM") else "/dev/shm"
var CTL_PATH := SHM.path_join("pcctl.json")
var HIT_PATH := SHM.path_join("pchit.json")
var STATUS_PATH := SHM.path_join("pcstatus.json")
var GNSS_PATH := SHM.path_join("gnss.json")
var GEOREF_PATH := SHM.path_join("georef.json")
var ZONES_PATH := SHM.path_join("pczones.json")
const CTL_S := 1.0 / 30.0
const HUD_S := 0.1
const TOOLS := ["scan", "paint", "measure", "anchor", "box", "path", "select", "zone"]
const MODELING := ["measure", "anchor", "box", "path"]
const ZONE_SHAPES := ["box", "outline", "cut below", "cut above"]
const ZONE_MODES := ["exclude", "include", "label"]
const HELP_START_S := 30.0
const HZ_PER_DEG := 0.0004   # horizon shift per degree of pitch (m at 0.7 m)

const HELP_TEXT := """POINT CLOUD PLANNER - HELP (Y: close)

RIGHT HAND (tools)
  bumper ...... click: toolbox button / place point
  left bumper . STILL: stereo capture (still.tiff) + next composited headset frame, no toolbox needed
  A ........... undo (last point, else last object)
  B ........... next tool
  X ........... stereo still (pair + pose + time + GNSS)
  Y ........... this help
  trigger hold  capture where you point (analog)
  grip ........ finish panel (box after 3 points) / path
  stick ↑↓ .... distance of the free point (orange)
  stick click . snap to scanned surface on/off (green)

LEFT HAND (toolbox above the controller)
  D-pad ↑ ..... scan: RECORD / STOP
  D-pad ↓ ..... hide / show cloud
  D-pad ← → ... point size
  View ........ toolbox on/off
  trigger hold  peek: hide cloud while held
  stick ↑↓ .... cloud opacity

TOOLS
  SCAN     fuse everything you look at
  PAINT    fuse only where you point (hold trigger);
           with scanning STOPPED the trigger does this in any tool
  MEASURE  2 clicks: distance, height diff, horizontal
  ANCHOR   1 click: named reference point
  BOX      4 clicks: edge, edge, width, height
           3 clicks + grip: 4 cm panel (solar panel)
  PATH     clicks + grip: conduit / cable run

DEPTH VIEW (toolbox): camera image drawn on its own measured
  depth (~3 Hz, world-locked). Where it matches the real world,
  the depth is right; gaps = no depth (SteamVR view shows).
  SELECT   hold trigger: brush over points (stick: size),
           then toolbox DELETE / KEEP ONLY / LABEL; grip clears
  ZONE     EXCLUDE / INCLUDE / LABEL region (toolbox):
           BOX 4 clicks, OUTLINE traced on the floor (hold
           trigger, like the playspace), CUT BELOW / ABOVE 1 click
           any include zone -> only what is inside is kept
           A undoes the last zone; zones never delete fused data
Clear cloud / objects / exit: toolbox, press twice.
Save (toolbox): cloud .ply + objects .json in captures/sessions"""

var xr_interface: OpenXRInterface
var origin: XROrigin3D
var camera: XRCamera3D
var cloud: CloudTiles
var model: SceneModel
var mat: ShaderMaterial
var status: Label3D
var status_low: Label3D
var horizon: Node3D
var hz_line: Node3D
var hz_label: Label3D
var help: Label3D
var tips := {}
var reticle: MeshInstance3D
var ret_mat: StandardMaterial3D
var palette: Palette
var pal_holder: Node3D
var passthrough: DepthPassthrough
var tag_layer: TagLayer                   # fiducial + QR labels (tools/tags.py)
var tags_on := true
var far_on := true                          # wide-baseline far-range depth (tools/farstereo.py)
var tag_accum := 0.0
var occlude := true                       # scene 90 % translucent where the real world (stereo depth) is closer
var occ_shown := false
var occ_ms := 0                           # last time an occlusion frame arrived
const PAL_OFFSET := Vector3(0.0, 0.09, -0.03)   # in the left controller frame (y up, -z forward)
const PAL_TILT_DEG := 25.0                         # face tilted up toward the eyes
var ctrl := {}
var rays := {}
var sizes := [0.003, 0.004, 0.006, 0.010, 0.016]
var size_idx := 2
var alpha := 0.55

var tool := "scan"
var scanning := true
var painting := false
var snap := true
var free_dist := 0.6
var cloud_shown := true
var peeking := false
var pts: Array = []          # clicked points of the object in progress
var cmd := ""
var cmd_seq := 0
var save_name := ""
var ctl_accum := 0.0
var hud_accum := 0.0
var hit := {}                # latest pchit.json
var hit_seq := -1
var hit_rx_ms := 0
var srv := {}
var gnss := {}
var georef := {}
var slow_accum := 0.0
var msg := ""
var msg_ms := 0
var sessions_dir := ""
var pal_hit = null           # palette hit of the right ray this frame
var analog := {}             # hand/action -> pressed state for float-derived buttons
var last_stills := -1
var still_pending_ms := -1
var profile := ""
var help_touched := false     # user toggled help -> no auto-hide
var input_log: Array = []     # last input events (shown in the help panel, printed to the log)
var sel_centers: Array = []   # SELECT brush dabs (map frame), pending until DELETE / KEEP ONLY / LABEL
var sel_radii: Array = []
var sel_strokes: Array = []   # index of the first dab of each stroke (A = undo stroke)
var brush_r := 0.06
var brushing := false
var zone_mode := "exclude"
var zone_shape := "box"
var label_idx := 1            # SceneModel.LABELS ("wall")
var tracing := false          # ZONE outline: trigger held, tracing on the floor
var zones_shown := true


func _ready() -> void:
	sessions_dir = OS.get_environment("PCVIEW_SESSIONS_DIR") if OS.has_environment("PCVIEW_SESSIONS_DIR") \
		else ProjectSettings.globalize_path("res://").path_join("../captures/sessions").simplify_path()
	DirAccess.make_dir_recursive_absolute(sessions_dir)
	_build_scene()
	_load_autosave()
	_write_zones()
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
		c.button_released.connect(_on_button_released.bind(hand))
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
		var tip := Label3D.new()
		tip.font_size = 24
		tip.pixel_size = 0.0004
		tip.outline_size = 6
		tip.no_depth_test = true
		tip.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		tip.modulate = Color(1.0, 0.95, 0.6)
		tip.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
		tip.position = Vector3(0.09 if hand == "right_hand" else -0.09, -0.02, 0.06)
		tip.text = "bumper: click  trigger: capture\nA undo  B next tool\nX still  Y help  grip: finish" if hand == "right_hand" \
			else "bumper: STILL (stereo capture)\nD-pad ↑ rec/stop  ↓ hide\n← → point size  View: toolbox"
		tip.visible = false
		c.add_child(tip)
		tips[hand] = tip
		origin.add_child(c)
	_set_ray("left_hand", 0.0)

	mat = ShaderMaterial.new()
	mat.shader = load("res://pc_points.gdshader")
	mat.set_shader_parameter("point_size", sizes[size_idx])
	mat.set_shader_parameter("alpha", alpha)
	cloud = CloudTiles.new(mat)
	cloud.path = SHM.path_join("pointcloud.bin")
	origin.add_child(cloud)
	passthrough = DepthPassthrough.new()      # depth-correct camera view (toolbox: DEPTH VIEW)
	passthrough.path = SHM.path_join("pcpass.bin")
	origin.add_child(passthrough)
	tag_layer = TagLayer.new()
	tag_layer.path = SHM.path_join("pctags.json")
	origin.add_child(tag_layer)

	model = SceneModel.new()
	model.changed.connect(_on_model_changed)
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

	# minimal HUD: one line high, one or two lines low, artificial horizon next to the top line
	status = _hud_label(Vector3(0.015, 0.205, -0.7), HORIZONTAL_ALIGNMENT_LEFT)
	status_low = _hud_label(Vector3(0, -0.215, -0.7), HORIZONTAL_ALIGNMENT_CENTER)
	status_low.pixel_size = 0.0006
	status_low.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	status_low.modulate = Color(0.75, 1.0, 0.8, 0.95)
	horizon = _build_horizon()
	horizon.scale = Vector3(2, 2, 2)
	horizon.position = Vector3(-0.045, 0.2, -0.7)
	camera.add_child(horizon)

	help = Label3D.new()
	help.text = HELP_TEXT
	help.font_size = 28
	help.pixel_size = 0.00022
	help.outline_size = 8
	help.vertical_alignment = VERTICAL_ALIGNMENT_TOP
	help.no_depth_test = true
	help.render_priority = 5
	help.outline_render_priority = 4
	help.modulate = Color(0.85, 0.95, 1.0)
	help.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	_show_help(true)
	get_tree().create_timer(HELP_START_S).timeout.connect(func(): if help.visible and not help_touched: _show_help(false))

	pal_holder = Node3D.new()
	origin.add_child(pal_holder)
	palette = Palette.new()
	pal_holder.add_child(palette)
	for i in range(TOOLS.size()):
		palette.add_button(TOOLS[i], TOOLS[i].to_upper(), i % 4, i / 4, "tool")
	palette.add_button("undo", "UNDO", 0, 2)
	palette.add_button("snap", "SNAP", 1, 2)
	palette.add_button("rec", "● REC", 2, 2)
	palette.add_button("still", "STILL", 3, 2)
	palette.add_button("save", "SAVE", 0, 3)
	palette.add_button("help", "HELP", 1, 3)
	palette.add_button("pass", "DEPTH VIEW", 2, 3)
	palette.add_button("hide", "HIDE CLOUD", 3, 3)
	palette.add_button("size-", "SIZE -", 0, 4)
	palette.add_button("size+", "SIZE +", 1, 4)
	palette.add_button("opacity", "OPACITY", 2, 4)
	palette.add_button("zone_show", "ZONES ON", 3, 4)
	palette.add_button("clear_cloud", "CLEAR CLOUD", 0, 5, "danger")
	palette.add_button("clear_objects", "CLEAR OBJ.", 1, 5, "danger")
	palette.add_button("occlude", "OCCLUDE", 2, 5)
	palette.add_button("exit", "EXIT", 3, 5, "danger")
	# per-tool row (SELECT / ZONE)
	palette.add_button("sel_delete", "DELETE", 0, 6)
	palette.add_button("sel_keep", "KEEP ONLY", 1, 6)
	palette.add_button("sel_label", "LABEL", 2, 6)
	palette.add_button("tags", "TAGS", 0, 7)
	palette.add_button("far", "FAR", 1, 7)
	palette.add_button("zone_mode", "EXCLUDE", 0, 6)
	palette.add_button("zone_shape", "BOX", 1, 6)
	palette.add_button("next_label", "NEXT LABEL", 3, 6)
	palette.pressed.connect(_on_palette)
	palette.add_child(help)                      # help + input diagnostics live next to the toolbox
	var pw := (palette.bg.mesh as QuadMesh).size.x
	help.position = Vector3(pw + 0.006, 0.02, 0.002)
	_sync_palette()


func _hud_label(pos: Vector3, align: int) -> Label3D:
	var l := Label3D.new()
	l.font_size = 24
	l.pixel_size = 0.00034
	l.outline_size = 6
	l.no_depth_test = true
	l.render_priority = 6
	l.outline_render_priority = 5
	l.modulate = Color(0.6, 1.0, 0.7, 0.9)
	l.horizontal_alignment = align
	l.position = pos
	camera.add_child(l)
	return l


func _hud_mat(c: Color) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = c
	m.no_depth_test = true
	m.render_priority = 6
	return m


func _bar(parent: Node3D, size: Vector2, c: Color, pos := Vector3.ZERO) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var q := QuadMesh.new()
	q.size = size
	mi.mesh = q
	mi.material_override = _hud_mat(c)
	mi.position = pos
	parent.add_child(mi)
	return mi


func _build_horizon() -> Node3D:
	## Small artificial horizon (4 cm at 0.7 m): sky/ground line rotates with roll, shifts with pitch;
	## fixed yellow aircraft marker; label with pitch / roll / heading of the HMD pose.
	var root := Node3D.new()
	_bar(root, Vector2(0.034, 0.024), Color(0.02, 0.05, 0.1, 0.45))
	hz_line = Node3D.new()
	root.add_child(hz_line)
	_bar(hz_line, Vector2(0.03, 0.0012), Color(0.5, 0.85, 1.0, 0.95), Vector3(0, 0, 0.0005))
	for k in [-2, -1, 1, 2]:                     # pitch ladder every 15 deg
		_bar(hz_line, Vector2(0.008 if k % 2 else 0.012, 0.0006), Color(0.5, 0.85, 1.0, 0.6), Vector3(0, k * HZ_PER_DEG * 15.0, 0.0005))
	_bar(root, Vector2(0.004, 0.0014), Color(1.0, 0.85, 0.2, 1.0), Vector3(-0.005, 0, 0.001))
	_bar(root, Vector2(0.004, 0.0014), Color(1.0, 0.85, 0.2, 1.0), Vector3(0.005, 0, 0.001))
	_bar(root, Vector2(0.0014, 0.0014), Color(1.0, 0.85, 0.2, 1.0), Vector3(0, 0, 0.001))
	hz_label = Label3D.new()
	hz_label.font_size = 20
	hz_label.pixel_size = 0.00030
	hz_label.outline_size = 5
	hz_label.no_depth_test = true
	hz_label.render_priority = 7
	hz_label.outline_render_priority = 6
	hz_label.modulate = Color(0.6, 0.9, 1.0, 0.9)
	hz_label.position = Vector3(0, -0.018, 0)
	root.add_child(hz_label)
	return root


func _update_horizon() -> void:
	var b := camera.global_transform.basis
	var fwd := -b.z
	var right := b.x
	var pitch := rad_to_deg(asin(clampf(fwd.y, -1.0, 1.0)))
	var roll := rad_to_deg(asin(clampf(right.y, -1.0, 1.0)))   # + = left ear down
	var heading := fposmod(rad_to_deg(atan2(fwd.x, -fwd.z)), 360.0)   # STAGE: 0 = -z (initial forward), clockwise
	hz_line.rotation = Vector3(0, 0, deg_to_rad(-roll))
	var off := clampf(-pitch * HZ_PER_DEG, -0.011, 0.011)
	hz_line.position = Vector3(-sin(deg_to_rad(-roll)) * off, cos(deg_to_rad(-roll)) * off, 0)
	if hud_accum == 0.0:                 # text at the HUD rate (see _process), line every frame
		hz_label.text = "P %+3.0f°  R %+3.0f°  H %03.0f°" % [pitch, roll, heading]


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


func _log_input(s: String) -> void:
	input_log.push_front("%.1f %s" % [Time.get_ticks_msec() / 1000.0, s])
	input_log.resize(mini(input_log.size(), 6))
	print("PCVIEW input ", s)


func _on_button_released(action: String, hand: String) -> void:
	_log_input("%s %s up" % [hand.left(1).to_upper(), action])
	if action in ["trigger_click", "grip_click"]:          # boolean fallback for trigger / grip
		var a := action.trim_suffix("_click")
		if analog.get(hand + "/" + a, false):
			analog[hand + "/" + a] = false
			_analog_released(hand, a)


func _on_button(action: String, hand: String) -> void:
	## Digital buttons. Trigger and grip: analog edges in _poll_analog, or the click actions as a fallback.
	_log_input("%s %s down" % [hand.left(1).to_upper(), action])
	if action in ["trigger_click", "grip_click"]:
		var a := action.trim_suffix("_click")
		if not analog.get(hand + "/" + a, false):
			analog[hand + "/" + a] = true
			_analog_pressed(hand, a)
		return
	if hand == "right_hand":
		match action:
			"ax_button":
				_undo()
			"by_button":
				_set_tool(TOOLS[(TOOLS.find(tool) + 1) % TOOLS.size()])
			"x_button":
				_still()
			"y_button":
				help_touched = true
				_show_help(not help.visible)
			"bumper":                  # the "digital trigger": click (toolbox buttons, place points)
				_primary()
			"menu_button":
				_toggle_palette()
			"primary_click":
				_toggle_snap()
	else:
		match action:
			"dpad_up", "by_button":          # by_button/ax_button: SteamVR Touch emulation (no native profile)
				_toggle_scan()
			"dpad_down", "ax_button":
				_toggle_cloud()
			"dpad_left":
				_point_size(-1)
			"dpad_right":
				_point_size(1)
			"view_button", "menu_button":
				_toggle_palette()
			"bumper":                     # dedicated STILL button (left index finger): stereo still + next composited frame, no toolbox
				_still()
	_write_ctl()


func _poll_analog() -> void:
	## Trigger / grip as buttons with hysteresis from the analog values.
	for hand in ["left_hand", "right_hand"]:
		var c: XRController3D = ctrl[hand]
		for a in ["trigger", "grip"]:
			var v := c.get_float(a)
			var key: String = hand + "/" + a
			var was: bool = analog.get(key, false)
			var now := v > 0.6 if not was else v > 0.35
			if now != was:
				analog[key] = now
				if now:
					_analog_pressed(hand, a)
				else:
					_analog_released(hand, a)


func _primary() -> void:
	## Right bumper (digital trigger): press the hovered toolbox button, else place a point with the tool.
	if pal_hit != null:
		var id := palette.press()
		_log_input("toolbox press '%s' (hovered '%s')" % [id, palette.hovered])
		_buzz("right_hand", 0.3 if id != "" else 0.15, 0.03)
	elif tool in MODELING:
		_place()
	elif tool == "select":
		_dab(true)
	elif tool == "zone":
		_zone_click()
	elif tool in ["scan", "paint"]:
		_say("bumper places points in MEASURE / ANCHOR / BOX / PATH; hold the analog trigger to capture")


func _analog_pressed(hand: String, a: String) -> void:
	if hand == "right_hand" and a == "trigger":
		if pal_hit != null:                     # also accept the analog trigger on the toolbox
			_primary()
			return
		if tool == "select":
			brushing = true                     # SELECT: sweep the brush while held
			_dab(true)
		elif tool == "zone" and zone_shape == "outline":
			tracing = true                      # ZONE outline: trace on the floor while held (like the playspace)
			pts.clear()
		else:
			painting = true                     # capture where you point (other tools)
	elif hand == "right_hand" and a == "grip":
		_finish()
	elif hand == "left_hand" and a == "trigger":
		peeking = true
		_apply_cloud_visibility()
	_write_ctl()


func _analog_released(hand: String, a: String) -> void:
	if hand == "right_hand" and a == "trigger":
		painting = false
		brushing = false
		if tracing:
			tracing = false
			_finish_outline()
	elif hand == "left_hand" and a == "trigger":
		peeking = false
		_apply_cloud_visibility()
	_write_ctl()


func _on_palette(id: String) -> void:
	if id in TOOLS:
		_set_tool(id)
		return
	match id:
		"undo":
			_undo()
		"snap":
			_toggle_snap()
		"rec":
			_toggle_scan()
		"still":
			_still()
		"save":
			_save()
		"help":
			help_touched = true
			_show_help(not help.visible)
		"hide":
			_toggle_cloud()
		"size-":
			_point_size(-1)
		"size+":
			_point_size(1)
		"opacity":
			alpha = 1.0 if alpha < 0.3 else (alpha - 0.25 if alpha > 0.5 else 0.2)
			mat.set_shader_parameter("alpha", alpha)
			_say("opacity %d %%" % int(alpha * 100))
		"clear_cloud":
			_send_cmd("reset")
			cloud.clear()
			_buzz("right_hand", 0.8, 0.15)
			_say("cloud cleared")
		"clear_objects":
			model.clear_all()
			pts.clear()
			model.set_preview("", [], null)
			_buzz("right_hand", 0.8, 0.15)
			_say("objects cleared")
		"pass":
			_toggle_passthrough()
		"far":
			far_on = not far_on
			_say("far range ON: wide-baseline depth from stored keyframes" if far_on else "far range off")
			_write_ctl()
			_sync_palette()
		"tags":
			tags_on = not tags_on
			tag_layer.visible = tags_on
			_say("tag labels ON: QR text + fiducial axes / IDs" if tags_on else "tag labels off")
			_write_ctl()
			_sync_palette()
		"occlude":
			occlude = not occlude
			_say("occlusion ON: the cloud fades where real objects are closer" if occlude else "occlusion off")
			_write_ctl()
			_sync_palette()
		"sel_delete":
			_apply_selection("exclude")
		"sel_keep":
			_apply_selection("include")
		"sel_label":
			_apply_selection("label")
		"next_label":
			label_idx = (label_idx + 1) % SceneModel.LABELS.size()
			_say("label " + _label_name().to_upper())
		"zone_mode":
			zone_mode = ZONE_MODES[(ZONE_MODES.find(zone_mode) + 1) % ZONE_MODES.size()]
			_say("zone mode " + zone_mode.to_upper())
		"zone_shape":
			zone_shape = ZONE_SHAPES[(ZONE_SHAPES.find(zone_shape) + 1) % ZONE_SHAPES.size()]
			pts.clear()
			model.set_preview("", [], null)
			_say("zone shape " + zone_shape.to_upper())
		"zone_show":
			zones_shown = not zones_shown
			model.set_zones_visible(zones_shown)
		"exit":
			_exit()
	_sync_palette()


func _sync_palette() -> void:
	for t in TOOLS:
		palette.set_active(t, t == tool)
	palette.set_active("rec", scanning, Palette.COLS["rec"])
	palette.set_text("rec", "● REC" if scanning else "■ STOPPED")
	palette.set_active("snap", snap)
	palette.set_active("hide", not cloud_shown)
	palette.set_text("hide", "SHOW CLOUD" if not cloud_shown else "HIDE CLOUD")
	palette.set_active("help", help.visible)
	palette.set_active("pass", passthrough.visible)
	palette.set_active("occlude", occlude)
	palette.set_active("tags", tags_on)
	palette.set_active("far", far_on)
	palette.set_active("zone_show", zones_shown)
	palette.set_text("zone_show", "ZONES ON" if zones_shown else "ZONES OFF")
	for id in ["sel_delete", "sel_keep", "sel_label"]:
		palette.set_shown(id, tool == "select")
	for id in ["zone_mode", "zone_shape"]:
		palette.set_shown(id, tool == "zone")
	palette.set_shown("next_label", tool == "select" or (tool == "zone" and zone_mode == "label"))
	palette.set_text("sel_label", "LABEL " + _label_name().to_upper())
	palette.set_text("zone_mode", zone_mode.to_upper() + (" " + _label_name().to_upper() if zone_mode == "label" else ""))
	palette.set_text("zone_shape", zone_shape.to_upper())
	palette.set_active("zone_mode", true, SceneModel.ZONE_COLS[zone_mode] * Color(0.6, 0.6, 0.6, 0.95) if zone_mode != "label" \
		else SceneModel.ZONE_COLS[_label_name()] * Color(0.6, 0.6, 0.6, 0.95))
	palette.set_title("%s   %s" % [tool.to_upper(), "● REC" if scanning else "■ STOPPED"])


func _set_tool(t: String) -> void:
	if pts.size() > 0 and tool in MODELING:
		_finish()
	brushing = false
	tracing = false
	tool = t
	painting = false
	pts.clear()
	model.set_preview("", [], null)
	_say("tool " + tool.to_upper())
	_sync_palette()


func _toggle_scan() -> void:
	scanning = not scanning
	_buzz("left_hand", 0.5, 0.06)
	_say("scan " + ("RECORDING" if scanning else "STOPPED"))
	_sync_palette()


func _toggle_cloud() -> void:
	cloud_shown = not cloud_shown
	_apply_cloud_visibility()
	_sync_palette()


func _apply_cloud_visibility() -> void:
	cloud.visible = cloud_shown and not peeking


func _toggle_snap() -> void:
	snap = not snap
	_say("surface snap " + ("on" if snap else "off"))
	_sync_palette()


func _toggle_palette() -> void:
	pal_holder.visible = not pal_holder.visible


func _point_size(d: int) -> void:
	size_idx = clampi(size_idx + d, 0, sizes.size() - 1)
	mat.set_shader_parameter("point_size", sizes[size_idx])
	_say("point size %.0f mm" % (sizes[size_idx] * 1000))


func _show_help(on: bool) -> void:
	help.visible = on
	for h in tips:
		tips[h].visible = on
	if palette:
		_sync_palette()


func _still() -> void:
	_send_cmd("still")
	still_pending_ms = Time.get_ticks_msec()
	_buzz("right_hand", 0.6, 0.05)
	_say("still: capturing ...")


func _exit() -> void:
	_autosave()
	_say("exit")
	var f := FileAccess.open(CTL_PATH, FileAccess.WRITE)     # tell the server the viewer is gone
	if f:
		f.store_string(JSON.stringify({"t_ms": 0, "scan": "off", "cmd": "", "cmd_seq": cmd_seq}))
		f.close()
	get_tree().quit()


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
	var c = _cursor()
	if c == null:
		return
	var p: Vector3 = c
	_buzz()
	match tool:
		"anchor":
			var o := model.add_anchor(p)
			_say("anchor %s" % o["name"])
		"measure":
			pts.append(p)
			if pts.size() == 2:
				var o := model.add_measure(pts[0], pts[1])
				_say("%s %s" % [o.get("name", "measure"), SceneModel.measure_text(pts[0], pts[1]).replace("\n", " ")])
				pts.clear()
		"box":
			pts.append(p)
			if pts.size() == 4:
				var o := model.add_box(pts)
				_say("box %s" % o.get("name", "failed (degenerate)"))
				pts.clear()
		"path":
			pts.append(p)


func _finish() -> void:
	if tool == "select":
		if not sel_centers.is_empty():
			_clear_selection()
			_say("selection cleared")
		return
	if tool == "zone" and zone_shape == "outline":
		_finish_outline()
		return
	if tool == "box" and pts.size() == 3:
		var o := model.add_box(pts)
		_say("panel %s" % o.get("name", "failed (degenerate)"))
	elif tool == "path" and pts.size() >= 2:
		var o := model.add_path(pts)
		_say("path %s %.2f m" % [o["name"], SceneModel.path_length(pts)])
	pts.clear()
	model.set_preview("", [], null)


func _undo() -> void:
	if tool == "select" and not sel_strokes.is_empty():
		var k: int = sel_strokes.pop_back()
		sel_centers.resize(k)
		sel_radii.resize(k)
		model.set_selection(sel_centers, sel_radii)
		_say("undo brush stroke")
	elif pts.size() > 0:
		pts.pop_back()
		_say("undo point")
	else:
		var o := model.remove_last()
		_say("removed %s" % o.get("name", "nothing"))
	_buzz("right_hand", 0.2, 0.02)


# ------------------------------------------------------------------ SELECT / ZONE

func _label_name() -> String:
	return SceneModel.LABELS[label_idx]


func _dab(new_stroke: bool) -> void:
	## SELECT brush: add a sphere at the cursor (scanned surface under the ray, else the free point).
	var c = _cursor()
	if c == null:
		return
	var p: Vector3 = c
	if not new_stroke and not sel_centers.is_empty() and (sel_centers[sel_centers.size() - 1] as Vector3).distance_to(p) < brush_r * 0.4:
		return
	if new_stroke:
		sel_strokes.append(sel_centers.size())
	sel_centers.append(p)
	sel_radii.append(brush_r)
	model.set_selection(sel_centers, sel_radii)


func _clear_selection() -> void:
	sel_centers.clear()
	sel_radii.clear()
	sel_strokes.clear()
	model.set_selection([], [])


func _apply_selection(mode: String) -> void:
	if sel_centers.is_empty():
		_say("nothing selected: hold the trigger and sweep over the points")
		return
	var z := {"mode": mode, "shape": "spheres", "centers": sel_centers.duplicate(), "radii": sel_radii.duplicate()}
	if mode == "label":
		z["label"] = _label_name()
	var o := model.add_zone(z)
	_clear_selection()
	_buzz("right_hand", 0.5, 0.06)
	_say(SceneModel.zone_text(o) + "   (A: undo)")


func _floor_point() -> Variant:
	## Ground point under the right ray for tracing zone outlines (STAGE): the scanned surface when there is a fresh
	## hit (works on sloped / uneven ground outdoors), else the SteamVR floor plane y = 0; null if neither.
	var c: XRController3D = ctrl.get("right_hand")
	if c == null or not c.get_has_tracking_data():
		return null
	if hit.get("hit", false) and Time.get_ticks_msec() - hit_rx_ms < 500:
		return c.global_transform.origin - c.global_transform.basis.z * float(hit["dist"])
	var o := c.global_transform.origin
	var d := -c.global_transform.basis.z
	if d.y > -0.02:
		return null
	var t := -o.y / d.y
	return o + d * t if t > 0.0 and t < 20.0 else null


func _zone_cursor() -> Variant:
	return _floor_point() if zone_shape == "outline" else _cursor()


func _zone_base() -> Dictionary:
	var z := {"mode": zone_mode}
	if zone_mode == "label":
		z["label"] = _label_name()
	return z


func _zone_click() -> void:
	var c = _zone_cursor()
	if c == null:
		_say("point at the floor" if zone_shape == "outline" else "no point under the ray")
		return
	var p: Vector3 = c
	_buzz()
	match zone_shape:
		"box":
			pts.append(p)
			if pts.size() == 4:
				var b := SceneModel.box_from_points(pts)
				pts.clear()
				if b.is_empty():
					_say("zone box failed (degenerate)")
					return
				var z := _zone_base()
				z.merge({"shape": "box", "center": b["center"], "axes": b["axes"], "size": b["size"]})
				_say(SceneModel.zone_text(model.add_zone(z)))
		"outline":
			pts.append(p)
		"cut below", "cut above":
			var below := zone_shape == "cut below"
			var o := model.add_zone({"mode": "exclude", "shape": "cut", "side": "below" if below else "above",
				"y": p.y - 0.02 if below else p.y + 0.02, "c": p})
			_say(SceneModel.zone_text(o) + "   (A: undo)")


func _finish_outline() -> void:
	if pts.size() >= 3:
		var z := _zone_base()
		z.merge({"shape": "prism", "points": pts.duplicate()})
		_say(SceneModel.zone_text(model.add_zone(z)) + "   (A: undo)")
	elif pts.size() > 0:
		_say("outline needs at least 3 points")
	pts.clear()
	model.set_preview("", [], null)


func _write_zones() -> void:
	_write_json(ZONES_PATH, {"seq": Time.get_unix_time_from_system(), "zones": model.zones()})


func _on_model_changed() -> void:
	_autosave()
	_write_zones()


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
	## auto: fuse everything in view; paint: fuse only the cone around the right aim ray; off.
	## The analog trigger captures where you point in every tool, also while scanning is stopped.
	if painting:
		return "paint" if (tool == "paint" or not scanning) else "auto"
	if not scanning or tool == "paint":
		return "off"
	return "auto"


func _write_ctl() -> void:
	var aim := []
	var c: XRController3D = ctrl.get("right_hand")
	if c != null and c.get_has_tracking_data():
		var t := c.global_transform        # XROrigin at identity -> STAGE space
		for r in range(3):
			aim.append_array([t.basis.x[r], t.basis.y[r], t.basis.z[r], t.origin[r]])
	var hands := []                         # both controllers (STAGE 3x4): the server keeps them and the hands out of the map
	for hn in ["left_hand", "right_hand"]:
		var hc: XRController3D = ctrl.get(hn)
		if hc != null and hc.get_has_tracking_data():
			var ht := hc.global_transform
			var m12 := []
			for r in range(3):
				m12.append_array([ht.basis.x[r], ht.basis.y[r], ht.basis.z[r], ht.origin[r]])
			hands.append(m12)
	var d := {"t_ms": Time.get_ticks_msec(), "scan": _scan_state(), "painting": painting and aim.size() == 12,
		"aim": aim, "hands": hands, "cmd": cmd, "cmd_seq": cmd_seq, "name": save_name,
		"passthrough": passthrough != null and passthrough.visible, "occlude": occlude, "tags": tags_on, "far": far_on}
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
	_buzz("right_hand", 0.6, 0.1)
	_say("saved " + save_name)


# ------------------------------------------------------------------ per frame

func _set_ray(hand: String, length: float, color := Color(0.3, 0.9, 1.0)) -> void:
	var r: MeshInstance3D = rays.get(hand)
	if r == null:
		return
	r.visible = length > 0.0
	(r.mesh as BoxMesh).size = Vector3(0.003, 0.003, maxf(length, 0.001))
	r.position = Vector3(0, 0, -length * 0.5)
	(r.material_override as StandardMaterial3D).albedo_color = color


func _place_palette() -> void:
	## Toolbox rigidly attached to the left controller (like Blocks): it follows position AND orientation, standing
	## 9 cm above the controller, its face toward the user, tilted back PAL_TILT_DEG so it reads when held in front.
	var lc: XRController3D = ctrl["left_hand"]
	if not lc.get_has_tracking_data():
		return
	pal_holder.global_transform = lc.global_transform * Transform3D(Basis(Vector3.RIGHT, deg_to_rad(-PAL_TILT_DEG)), PAL_OFFSET)
	var size := (palette.bg.mesh as QuadMesh).size
	palette.position = Vector3(-size.x * 0.5, size.y * 0.5, 0)


func _toggle_passthrough() -> void:
	passthrough.visible = not passthrough.visible
	_say("depth view ON: camera image on its measured depth" if passthrough.visible else "depth view off")
	_write_ctl()
	_sync_palette()


func _process(delta: float) -> void:
	if passthrough.visible or occlude:
		if passthrough.poll():
			occ_ms = Time.get_ticks_msec()
			passthrough.apply_occlusion(mat)
	tag_accum += delta
	if tags_on and tag_accum >= 0.25:
		tag_accum = 0.0
		tag_layer.poll()
	var occ_live := occlude and Time.get_ticks_msec() - occ_ms < 1000
	if occ_live != occ_shown:
		occ_shown = occ_live
		mat.set_shader_parameter("occ_on", 1.0 if occ_live else 0.0)
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
		var n := int(srv.get("stills", 0))
		if still_pending_ms >= 0 and n != last_stills and last_stills >= 0:
			_say("still saved: " + String(srv.get("still", "")).get_file())
			_buzz("right_hand", 0.3, 0.03)
			still_pending_ms = -1
		elif still_pending_ms >= 0 and Time.get_ticks_msec() - still_pending_ms > 5000:
			_say("still: no answer from the server (is pcserver running?)")
			still_pending_ms = -1
		last_stills = n
		var tr = XRServer.get_tracker(&"right_hand")
		if tr and tr is XRControllerTracker and String(tr.profile) != profile:
			profile = String(tr.profile)
			print("PCVIEW controller profile: ", profile)

	_place_palette()
	_poll_analog()

	var rc: XRController3D = ctrl.get("right_hand")
	var lc: XRController3D = ctrl.get("left_hand")
	if rc:
		var st := rc.get_vector2("primary")
		if abs(st.y) > 0.3 and tool == "select":
			brush_r = clampf(brush_r * (1.0 + st.y * delta * 1.5), 0.01, 0.6)
		elif abs(st.y) > 0.3:
			free_dist = clampf(free_dist * (1.0 + st.y * delta * 1.5), 0.15, 8.0)
	if lc:
		var st := lc.get_vector2("primary")
		if abs(st.y) > 0.3:
			alpha = clampf(alpha + st.y * delta * 0.8, 0.1, 1.0)
			mat.set_shader_parameter("alpha", alpha)

	# right ray: toolbox first, then the tool cursor
	pal_hit = null
	if rc and rc.get_has_tracking_data():
		var t := rc.global_transform
		pal_hit = palette.ray_hit(t.origin, -t.basis.z)
	var modeling: bool = tool in MODELING or tool == "select" or tool == "zone"
	if brushing:
		_dab(false)
	if tracing:
		var fp = _floor_point()
		if fp != null:
			var mp: Vector3 = fp
			if pts.is_empty() or (pts[pts.size() - 1] as Vector3).distance_to(mp) > 0.05:
				pts.append(mp)
	reticle.scale = Vector3.ONE * (brush_r / 0.008 if tool == "select" and pal_hit == null else 1.0)
	if pal_hit != null:
		reticle.visible = true
		reticle.position = pal_hit["point"]
		ret_mat.albedo_color = Color(0.5, 0.8, 1.0, 0.95)
		_set_ray("right_hand", pal_hit["dist"], Color(0.5, 0.8, 1.0))
		model.set_preview("", [], null)
	else:
		var cur = (_zone_cursor() if tool == "zone" else _cursor()) if modeling else null
		if rc and cur != null:
			var on_surface: bool = snap and hit.get("hit", false) and Time.get_ticks_msec() - hit_rx_ms < 500
			reticle.visible = true
			reticle.position = cur
			ret_mat.albedo_color = Color(0.3, 1.0, 0.4, 0.95) if on_surface else Color(1.0, 0.6, 0.1, 0.95)
			_set_ray("right_hand", (cur - rc.global_transform.origin).length(), Color(0.6, 1.0, 0.6) if on_surface else Color(1.0, 0.7, 0.3))
			if tool in ["box", "path", "measure"]:
				model.set_preview(tool, pts, cur)
			elif tool == "zone" and zone_shape in ["box", "outline"]:
				model.set_preview(zone_shape, pts, cur)
		else:
			reticle.visible = false
			if tool == "paint":
				_set_ray("right_hand", 2.0 if painting else 0.3, Color(1.0, 0.25, 0.2) if painting else Color(1.0, 0.6, 0.1))
			else:
				_set_ray("right_hand", 0.3)

	hud_accum += delta
	if hud_accum >= HUD_S:
		hud_accum = 0.0
	_update_horizon()
	if hud_accum != 0.0:                  # Label3D text changes rebuild meshes: update the text at 10 Hz, not per frame
		return
	if help.visible:
		var r: XRController3D = ctrl["right_hand"]
		var l: XRController3D = ctrl["left_hand"]
		help.text = HELP_TEXT + "\n\nINPUT  R trig %.2f grip %.2f   L trig %.2f grip %.2f   profile %s\n%s" % [
			r.get_float("trigger"), r.get_float("grip"), l.get_float("trigger"), l.get_float("grip"), profile.get_file(),
			"\n".join(input_log)]
	_update_status()


func _hint() -> String:
	match tool:
		"scan":
			return "look around to scan   B: next tool"
		"paint":
			return "hold trigger to capture where you point"
		"measure":
			return "bumper: %s   A: undo" % ["start point", "end point"][mini(pts.size(), 1)]
		"anchor":
			return "bumper: place anchor   A: undo"
		"box":
			var steps := ["corner 1", "corner 2 (first edge)", "width (or grip: panel)", "height"]
			return "bumper: %s   A: undo" % steps[mini(pts.size(), 3)]
		"path":
			return "bumper: add point (%d)   grip: finish   A: undo" % pts.size()
		"select":
			return "hold trigger: brush (%d cm, stick ↑↓)   %d dabs   toolbox: DELETE / KEEP ONLY / LABEL   grip: clear   A: undo stroke" % [
				roundi(brush_r * 100), sel_centers.size()]
		"zone":
			match zone_shape:
				"box":
					var steps := ["corner 1", "corner 2 (first edge)", "width", "height"]
					return "%s box   bumper: %s   A: undo" % [zone_mode.to_upper(), steps[mini(pts.size(), 3)]]
				"outline":
					return "%s outline   hold trigger: trace on the floor (or bumper: corners, grip: close)   %d pts" % [zone_mode.to_upper(), pts.size()]
				_:
					return "bumper on the floor/ceiling: remove everything %s it   A: undo" % ("below" if zone_shape == "cut below" else "above")
	return ""


func _update_status() -> void:
	status.text = "%s  %s%s   %.1f fr/s  %d fps  %dk pts%s" % ["● REC" if scanning else "■ STOP", tool.to_upper(),
		" ●" if painting else "", float(srv.get("fps", 0.0)), Engine.get_frames_per_second(), cloud.total_points / 1000,
		"" if cloud.visible else "  (hidden)"]
	# always-visible time + camera frame stamp, to find a moment of a composited capture in the recording
	# (f = arcgrab frame number = the "frame" field of frames.jsonl / poses.jsonl)
	var ut := Time.get_unix_time_from_system()
	var dt := Time.get_datetime_dict_from_unix_time(int(ut) + int(Time.get_time_zone_from_system()["bias"]) * 60)
	status.text += "   %02d:%02d:%02d.%d  f%d" % [dt["hour"], dt["minute"], dt["second"], int(fmod(ut, 1.0) * 10), int(srv.get("frame", -1))]
	if String(srv.get("origin", "")) != "":
		status.text += "   " + String(srv["origin"])
	if far_on and String(srv.get("far", "")) != "":
		status.text += "   " + String(srv["far"])
	if not gnss.is_empty():
		status.text += "   GNSS %s %.2f m%s" % [gnss.get("fix", "?"), float(gnss.get("hacc", -1)),
			"  geo %.2f" % float(georef.get("rms", 0)) if georef.get("ok", false) else ""]
	var low := []
	if Time.get_ticks_msec() - msg_ms < 4000:
		low.append(msg)
	if tool != "scan":
		low.append(_hint())
	if profile != "" and not "frame_controller" in profile:
		low.append("controller profile %s: X/Y/D-pad may not work" % profile.get_file())
	status_low.text = "\n".join(low)
