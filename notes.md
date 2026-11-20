Your point-cloud project has the same bug. I haven't touched it, but the fix there is two lines:
- viewer/project.godot: openxr/foveation_level=0 and openxr/foveation_dynamic=false
- viewer/start_vr.gd: remove the vp.vrs_mode = Viewport.VRS_XR branch

Keep submit_depth_buffer=true in both projects; turning it off made anchoring worse.
