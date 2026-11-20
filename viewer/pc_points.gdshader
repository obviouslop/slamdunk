// Camera-facing square per MultiMesh instance (one point), size in metres, unshaded vertex colour.
// Semi-transparent (no depth write) so the passthrough stays readable; points fade out near the eyes
// (they would otherwise cover large parts of the view).
shader_type spatial;
render_mode unshaded, cull_disabled, blend_mix, depth_draw_never;

uniform float point_size = 0.006;
uniform float alpha = 0.55;
uniform float near_fade_start = 0.25;   // fully invisible closer than this (m)
uniform float near_fade_end = 0.60;     // fully visible beyond this (m)
// Occlusion: where the stereo depth (coarse grid, published by the server) says a real surface is closer to the camera than
// this point, the point becomes occ_alpha times as visible (hands, furniture in front of the virtual scene stay readable).
uniform float occ_on = 0.0;
uniform float occ_alpha = 0.1;
uniform float occ_margin = 0.08;
uniform sampler2D occ_depth : filter_nearest;
uniform mat4 occ_rect_from_stage;
uniform vec4 occ_intr = vec4(1.0, 0.0, 0.0, 1.0);   // f, cx, cy, n of the rectified image
uniform float occ_block = 8.0;
varying vec3 col;
varying float vis;

void vertex() {
	// MODEL_MATRIX includes the instance transform (= point position); QuadMesh VERTEX is in [-0.5, 0.5]
	vec4 c = VIEW_MATRIX * MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0);
	vis = clamp((-c.z - near_fade_start) / max(near_fade_end - near_fade_start, 1e-3), 0.0, 1.0);
	if (occ_on > 0.5) {
		vec3 pr = (occ_rect_from_stage * MODEL_MATRIX * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
		if (pr.z > 0.1) {
			vec2 g = (occ_intr.x * pr.xy / pr.z + occ_intr.yz) / occ_block;
			ivec2 sz = textureSize(occ_depth, 0);
			if (g.x >= 0.0 && g.y >= 0.0 && g.x < float(sz.x) && g.y < float(sz.y)) {
				float z = texelFetch(occ_depth, ivec2(g), 0).r;
				if (z > 0.0 && z < pr.z - occ_margin) vis *= occ_alpha;
			}
		}
	}
	c.xy += VERTEX.xy * point_size;
	POSITION = PROJECTION_MATRIX * c;
	col = COLOR.rgb;
}

void fragment() {
	ALBEDO = col;
	ALPHA = alpha * vis;
}
