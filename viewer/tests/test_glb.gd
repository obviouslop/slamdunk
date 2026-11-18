extends SceneTree
## godot --headless --path viewer --xr-mode off -s res://tests/test_glb.gd -- file.glb
func _initialize() -> void:
	var path: String = OS.get_cmdline_user_args()[0]
	var doc := GLTFDocument.new()
	var st := GLTFState.new()
	var err := doc.append_from_file(path, st)
	print("append_from_file: ", error_string(err))
	var root := doc.generate_scene(st)
	_dump(root, 0)
	quit(0 if err == OK else 1)

func _dump(n: Node, d: int) -> void:
	var extra := ""
	if n is Node3D:
		extra = " pos %s" % [(n as Node3D).global_position if n.is_inside_tree() else (n as Node3D).position]
	if n is MeshInstance3D and (n as MeshInstance3D).mesh:
		var m: Mesh = (n as MeshInstance3D).mesh
		extra += " surfaces %d prim %d verts %d" % [m.get_surface_count(), m.surface_get_primitive_type(0), m.surface_get_arrays(0)[Mesh.ARRAY_VERTEX].size()]
	print("  ".repeat(d), n.name, " (", n.get_class(), ")", extra)
	for c in n.get_children():
		_dump(c, d + 1)
