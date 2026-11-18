class_name Palette
extends Node3D
## Toolbox panel that floats above the left controller (idea from Blocks / Open Blocks: palette on the
## non-dominant hand, point at it with the dominant hand). Buttons are flat quads with labels in a grid;
## hit-testing is a ray-plane intersection in the panel's local frame (no physics needed).
## Destructive buttons ("danger") need a second press within CONFIRM_S ("sure?" state).

signal pressed(id: String)

const CELL := Vector2(0.062, 0.030)
const GAP := 0.004
const CONFIRM_S := 3.0
const COLS := {
	"tool": Color(0.16, 0.30, 0.42, 0.85), "action": Color(0.20, 0.24, 0.30, 0.85),
	"danger": Color(0.45, 0.12, 0.12, 0.85), "hover": Color(0.35, 0.60, 0.80, 0.95),
	"active": Color(0.15, 0.62, 0.35, 0.95), "confirm": Color(0.95, 0.25, 0.15, 0.98), "rec": Color(0.85, 0.10, 0.10, 0.95),
	"off": Color(0.12, 0.12, 0.12, 0.6)}

var buttons := {}            # id -> {quad, label, mat, kind, rect: Rect2, text, active, enabled, color_on}
var hovered := ""
var confirm_id := ""
var confirm_ms := 0
var title: Label3D
var bg: MeshInstance3D
var rows := 0
var cols := 4


func _init() -> void:
	bg = MeshInstance3D.new()
	bg.mesh = QuadMesh.new()
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.albedo_color = Color(0.03, 0.04, 0.06, 0.55)
	m.no_depth_test = true
	m.render_priority = 1
	bg.material_override = m
	add_child(bg)
	title = Label3D.new()
	title.pixel_size = 0.00035
	title.font_size = 32
	title.outline_size = 8
	title.no_depth_test = true
	title.render_priority = 3
	title.outline_render_priority = 2
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_LEFT
	add_child(title)


func add_button(id: String, text: String, col: int, row: int, kind := "action", span := 1) -> void:
	var q := MeshInstance3D.new()
	var qm := QuadMesh.new()
	qm.size = Vector2(CELL.x * span + GAP * (span - 1), CELL.y)
	q.mesh = qm
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	m.no_depth_test = true
	m.render_priority = 2
	q.material_override = m
	var x := (col + span * 0.5) * (CELL.x + GAP)
	var y := -(row + 0.5) * (CELL.y + GAP)
	q.position = Vector3(x, y, 0.001)
	add_child(q)
	var l := Label3D.new()
	l.text = text
	l.pixel_size = 0.00028
	l.font_size = 30
	l.outline_size = 6
	l.no_depth_test = true
	l.render_priority = 4
	l.outline_render_priority = 3
	l.position = Vector3(x, y, 0.002)
	add_child(l)
	var half := qm.size * 0.5
	buttons[id] = {"quad": q, "label": l, "mat": m, "kind": kind, "text": text, "active": false, "enabled": true,
		"rect": Rect2(Vector2(x, y) - half, qm.size), "on_color": COLS["active"]}
	rows = maxi(rows, row + 1)
	cols = maxi(cols, col + span)
	_layout_bg()
	_refresh(id)


func _layout_bg() -> void:
	var w := cols * (CELL.x + GAP) + GAP
	var h := rows * (CELL.y + GAP) + GAP + 0.022
	(bg.mesh as QuadMesh).size = Vector2(w, h)
	bg.position = Vector3(w * 0.5 - GAP, -h * 0.5 + 0.022, 0)
	title.position = Vector3(0.0, 0.011, 0.002)


func set_title(t: String) -> void:
	title.text = t


func set_active(id: String, on: bool, color = null) -> void:
	if not buttons.has(id):
		return
	buttons[id]["active"] = on
	if color != null:
		buttons[id]["on_color"] = color
	_refresh(id)


func set_shown(id: String, on: bool) -> void:
	## Per-tool buttons: hidden buttons are not drawn and not hit (their cell stays reserved).
	if not buttons.has(id) or buttons[id]["enabled"] == on:
		return
	buttons[id]["enabled"] = on
	buttons[id]["quad"].visible = on
	buttons[id]["label"].visible = on
	if not on and hovered == id:
		hovered = ""
	if not on and confirm_id == id:
		confirm_id = ""


func set_text(id: String, t: String) -> void:
	if buttons.has(id) and buttons[id]["text"] != t:
		buttons[id]["text"] = t
		if confirm_id != id:
			buttons[id]["label"].text = t


func _refresh(id: String) -> void:
	var b: Dictionary = buttons[id]
	var c: Color = COLS[b["kind"]]
	if b["active"]:
		c = b["on_color"]
	if id == confirm_id:
		var blink := 0.75 + 0.25 * sin(Time.get_ticks_msec() * 0.015)
		c = COLS["confirm"] * Color(blink, blink, blink, 1.0)
	elif id == hovered:
		c = c.lerp(COLS["hover"], 0.6)
	(b["mat"] as StandardMaterial3D).albedo_color = c
	b["label"].text = "sure? press again" if id == confirm_id else b["text"]
	b["quad"].scale = Vector3.ONE * (1.06 if id == hovered else 1.0)


func ray_hit(origin: Vector3, dir: Vector3) -> Variant:
	## -> {"id", "point", "dist"} if the ray hits a button of the (visible) panel, else null; updates hover.
	var prev := hovered
	hovered = ""
	var res = null
	if is_visible_in_tree():
		var inv := global_transform.affine_inverse()
		var o := inv * origin
		var d := inv.basis * dir
		if abs(d.z) > 1e-6:
			var t := -o.z / d.z
			if t > 0.0 and t < 1.5:
				var p := o + d * t
				for id in buttons:
					if buttons[id]["enabled"] and (buttons[id]["rect"] as Rect2).has_point(Vector2(p.x, p.y)):
						hovered = id
						res = {"id": id, "point": global_transform * p, "dist": t * dir.length()}
						break
				if res == null:
					var r := Rect2(Vector2(bg.position.x, bg.position.y) - (bg.mesh as QuadMesh).size * 0.5, (bg.mesh as QuadMesh).size)
					if r.has_point(Vector2(p.x, p.y)):
						res = {"id": "", "point": global_transform * p, "dist": t * dir.length()}
	if prev != hovered:
		if buttons.has(prev):
			_refresh(prev)
		if buttons.has(hovered):
			_refresh(hovered)
	return res


func press() -> String:
	## Trigger while hovering: returns the activated id ("" if none / waiting for confirmation).
	if hovered == "":
		return ""
	var id := hovered
	if buttons[id]["kind"] == "danger":
		if confirm_id != id:
			var old := confirm_id
			confirm_id = id
			confirm_ms = Time.get_ticks_msec()
			if buttons.has(old):
				_refresh(old)
			_refresh(id)
			return ""
		confirm_id = ""
		_refresh(id)
	pressed.emit(id)
	return id


func _process(_delta: float) -> void:
	if confirm_id != "":
		if Time.get_ticks_msec() - confirm_ms > CONFIRM_S * 1000.0:
			var id := confirm_id
			confirm_id = ""
			_refresh(id)
		else:
			_refresh(confirm_id)
