class_name DepthPassthrough
extends MeshInstance3D
## Depth-correct passthrough: the latest rectified left camera image drawn on its own measured depth.
## The server (tools/ptpub.py) writes /dev/shm/pcpass.bin: colour (RGB8), a coarse depth grid (float32 m) and the
## STAGE-from-camera pose of that exposure. A grid mesh with one vertex per depth cell is displaced in the vertex
## shader (unproject with f/cx/cy, then the pose), so the image is world-locked on the real surfaces. Cells without
## depth and triangles across depth edges are discarded -> SteamVR's own passthrough shows through there.
## Must be a child of the XROrigin (STAGE frame), with identity transform.

const PATH := "/dev/shm/pcpass.bin"
const HDR := 128
const SHADER := """
shader_type spatial;
render_mode unshaded, cull_disabled, depth_draw_opaque;
uniform sampler2D depth_tex : filter_nearest;
uniform sampler2D colour_tex : source_color, filter_linear;
uniform mat4 stage_from_rect;
uniform vec4 intr;            // f, cx, cy, n (rectified pixels)
uniform float block = 8.0;    // depth pixels per grid cell
uniform float edge = 0.08;    // max relative depth jump between neighbouring cells
varying float bad;
varying vec2 cuv;

void vertex() {
	ivec2 ij = ivec2(UV);
	ivec2 sz = textureSize(depth_tex, 0);
	float z = texelFetch(depth_tex, ij, 0).r;
	float zmin = z;
	float zmax = z;
	for (int k = 0; k < 4; k++) {
		ivec2 o = k == 0 ? ivec2(1, 0) : k == 1 ? ivec2(-1, 0) : k == 2 ? ivec2(0, 1) : ivec2(0, -1);
		ivec2 q = clamp(ij + o, ivec2(0), sz - 1);
		float zn = texelFetch(depth_tex, q, 0).r;
		if (zn > 0.0) { zmin = min(zmin, zn); zmax = max(zmax, zn); }
	}
	bad = (z <= 0.0 || zmax > zmin * (1.0 + edge)) ? 1.0 : 0.0;
	float zz = z > 0.0 ? z : 2.0;
	vec2 px = (vec2(ij) + 0.5) * block - 0.5;
	vec3 pc = vec3((px.x - intr.y) / intr.x * zz, (px.y - intr.z) / intr.x * zz, zz);
	VERTEX = (stage_from_rect * vec4(pc, 1.0)).xyz;
	cuv = (px + 0.5) / intr.w;
}

void fragment() {
	if (bad > 0.001) discard;
	ALBEDO = texture(colour_tex, cuv).rgb;
}
"""

var seq := -1
var frame := -1
var ts := 0.0
var grid := Vector2i.ZERO
var mat: ShaderMaterial
var depth_img: Image
var colour_img: Image
var depth_tex: ImageTexture
var colour_tex: ImageTexture
var path := PATH
var last_read_ms := 0


func _init() -> void:
	mat = ShaderMaterial.new()
	mat.shader = Shader.new()
	mat.shader.code = SHADER
	material_override = mat
	cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	extra_cull_margin = 16384.0         # vertices are moved in the shader; never frustum-cull the mesh
	visible = false


func _build_grid(w: int, h: int) -> void:
	var v := PackedVector3Array()
	var uv := PackedVector2Array()
	v.resize(w * h)
	uv.resize(w * h)
	for i in range(h):
		for j in range(w):
			v[i * w + j] = Vector3.ZERO
			uv[i * w + j] = Vector2(j + 0.25, i + 0.25)      # integer cell index (+ margin against rounding)
	var idx := PackedInt32Array()
	for i in range(h - 1):
		for j in range(w - 1):
			var a := i * w + j
			idx.append_array([a, a + 1, a + w, a + 1, a + w + 1, a + w])
	var arr := []
	arr.resize(Mesh.ARRAY_MAX)
	arr[Mesh.ARRAY_VERTEX] = v
	arr[Mesh.ARRAY_TEX_UV] = uv
	arr[Mesh.ARRAY_INDEX] = idx
	var m := ArrayMesh.new()
	m.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arr)
	mesh = m
	grid = Vector2i(w, h)


func poll() -> bool:
	## Load a new frame if the server wrote one. -> true when updated.
	if not FileAccess.file_exists(path):
		return false
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return false
	var hdr := f.get_buffer(HDR)
	if hdr.size() < HDR or hdr.slice(0, 4).get_string_from_ascii() != "PCPT":
		return false
	var s := hdr.decode_u32(4)
	if s == seq:
		return false
	var wc := hdr.decode_u32(8)
	var hc := hdr.decode_u32(12)
	var wd := hdr.decode_u32(16)
	var hd := hdr.decode_u32(20)
	var body := f.get_buffer(wc * hc * 3 + wd * hd * 4)
	if body.size() < wc * hc * 3 + wd * hd * 4:
		return false
	return apply_frame(hdr, body)


func apply_frame(hdr: PackedByteArray, body: PackedByteArray) -> bool:
	var wc := hdr.decode_u32(8)
	var hc := hdr.decode_u32(12)
	var wd := hdr.decode_u32(16)
	var hd := hdr.decode_u32(20)
	var fl := []
	for k in range(4 + 12):
		fl.append(hdr.decode_float(24 + 4 * k))
	var cimg := Image.create_from_data(wc, hc, false, Image.FORMAT_RGB8, body.slice(0, wc * hc * 3))
	var dimg := Image.create_from_data(wd, hd, false, Image.FORMAT_RF, body.slice(wc * hc * 3, wc * hc * 3 + wd * hd * 4))
	if colour_tex == null or colour_img.get_size() != cimg.get_size():
		colour_tex = ImageTexture.create_from_image(cimg)
	else:
		colour_tex.update(cimg)
	if depth_tex == null or depth_img.get_size() != dimg.get_size():
		depth_tex = ImageTexture.create_from_image(dimg)
	else:
		depth_tex.update(dimg)
	colour_img = cimg
	depth_img = dimg
	if grid != Vector2i(wd, hd):
		_build_grid(wd, hd)
	# rows of the 3x4 (OpenCV camera axes -> STAGE); Godot Basis(columns)
	var r := []
	for k in range(12):
		r.append(fl[4 + k])
	var b := Basis(Vector3(r[0], r[4], r[8]), Vector3(r[1], r[5], r[9]), Vector3(r[2], r[6], r[10]))
	var t := Transform3D(b, Vector3(r[3], r[7], r[11]))
	mat.set_shader_parameter("stage_from_rect", Projection(t))
	mat.set_shader_parameter("intr", Vector4(fl[0], fl[1], fl[2], fl[3]))
	mat.set_shader_parameter("block", hdr.decode_float(100))
	mat.set_shader_parameter("depth_tex", depth_tex)
	mat.set_shader_parameter("colour_tex", colour_tex)
	seq = hdr.decode_u32(4)
	ts = hdr.decode_double(88)
	frame = hdr.decode_u32(96)
	return true


func apply_occlusion(m: ShaderMaterial) -> void:
	## Hand the newest depth grid to a material that has the occ_* uniforms (the cloud shader).
	m.set_shader_parameter("occ_depth", depth_tex)
	m.set_shader_parameter("occ_rect_from_stage", Projection(Transform3D(mat.get_shader_parameter("stage_from_rect")).affine_inverse()))
	m.set_shader_parameter("occ_intr", mat.get_shader_parameter("intr"))
	m.set_shader_parameter("occ_block", mat.get_shader_parameter("block"))


func occludes(p: Vector3, margin := 0.08) -> bool:
	## CPU copy of the cloud shader's test: is the measured real surface in front of STAGE point p (towards the camera)?
	if depth_img == null:
		return false
	var intr: Vector4 = mat.get_shader_parameter("intr")
	var blk: float = mat.get_shader_parameter("block")
	var pr := Transform3D(mat.get_shader_parameter("stage_from_rect")).affine_inverse() * p
	if pr.z < 0.1:
		return false
	var g := Vector2(intr.x * pr.x / pr.z + intr.y, intr.x * pr.y / pr.z + intr.z) / blk
	var gi := Vector2i(int(floor(g.x)), int(floor(g.y)))
	if gi.x < 0 or gi.y < 0 or gi.x >= grid.x or gi.y >= grid.y:
		return false
	for o in [Vector2i(0, 0), Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:    # same eroded test as the shader
		var q := Vector2i(clampi(gi.x + o.x, 0, grid.x - 1), clampi(gi.y + o.y, 0, grid.y - 1))
		var z := depth_img.get_pixel(q.x, q.y).r
		if not (z > 0.0 and z < pr.z - margin):
			return false
	return true


func cell_point(i: int, j: int) -> Variant:
	## CPU copy of the vertex shader (tests / picking): STAGE position of grid cell (row i, col j), null if invalid.
	if depth_img == null:
		return null
	var z := depth_img.get_pixel(j, i).r
	if z <= 0.0:
		return null
	var intr: Vector4 = mat.get_shader_parameter("intr")
	var blk: float = mat.get_shader_parameter("block")
	var px := (Vector2(j, i) + Vector2(0.5, 0.5)) * blk - Vector2(0.5, 0.5)
	var pc := Vector3((px.x - intr.y) / intr.x * z, (px.y - intr.z) / intr.x * z, z)
	var t := Transform3D(mat.get_shader_parameter("stage_from_rect"))
	return t * pc
