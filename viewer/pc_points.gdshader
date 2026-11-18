// Camera-facing square per MultiMesh instance (one point), size in metres, unshaded vertex colour.
// Semi-transparent (no depth write) so the passthrough stays readable; points fade out near the eyes
// (they would otherwise cover large parts of the view).
shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never;

uniform float point_size = 0.006;
uniform float alpha = 0.55;
uniform float near_fade_start = 0.25;   // fully invisible closer than this (m)
uniform float near_fade_end = 0.60;     // fully visible beyond this (m)
varying vec3 col;
varying float vis;

void vertex() {
	// MODEL_MATRIX includes the instance transform (= point position); QuadMesh VERTEX is in [-0.5, 0.5]
	vec4 c = VIEW_MATRIX * MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0);
	vis = clamp((-c.z - near_fade_start) / max(near_fade_end - near_fade_start, 1e-3), 0.0, 1.0);
	c.xy += VERTEX.xy * point_size;
	POSITION = PROJECTION_MATRIX * c;
	col = COLOR.rgb;
}

void fragment() {
	ALBEDO = col;
	ALPHA = alpha * vis;
}
