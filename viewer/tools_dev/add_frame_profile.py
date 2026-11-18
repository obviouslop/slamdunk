#!/usr/bin/env python3
"""Add Steam Frame controller actions + the native OpenXR profile /interaction_profiles/valve/frame_controller_valve
(XR_VALVE_frame_controller_interaction, known to Godot 4.7 and SteamVR) to viewer/openxr_action_map.tres.
Idempotent. Right controller: A B X Y, menu, trigger, squeeze, thumbstick, bumper. Left: D-pad, view, same analog."""
import re, sys, os
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "openxr_action_map.tres")
t = open(fn).read()
if "frame_controller_valve" in t:
    print("already present"); sys.exit()
H = "/user/hand/left", "/user/hand/right"
new_actions = [("x_button", "X button"), ("y_button", "Y button"), ("dpad_up", "D-pad up"), ("dpad_down", "D-pad down"),
               ("dpad_left", "D-pad left"), ("dpad_right", "D-pad right"), ("bumper", "Bumper"), ("view_button", "View button")]
acts = dict((n, i) for i, n in re.findall(r'\[sub_resource type="OpenXRAction" id="([^"]+)"\]\s*resource_name = "([^"]+)"', t))
blocks = []
for n, loc in new_actions:
    aid = f"OpenXRAction_fr_{n}"; acts[n] = aid
    blocks.append(f'[sub_resource type="OpenXRAction" id="{aid}"]\nresource_name = "{n}"\nlocalized_name = "{loc}"\naction_type = 0\n'
                  f'toplevel_paths = PackedStringArray("{H[0]}", "{H[1]}")\n')
# register in the action set
t = re.sub(r'(\[sub_resource type="OpenXRActionSet"[^\]]*\][^\[]*?actions = \[)([^\]]*)\]',
           lambda m: m.group(1) + m.group(2) + "".join(f', SubResource("OpenXRAction_fr_{n}")' for n, _ in new_actions) + "]", t, count=1)
both = lambda p: [f"/user/hand/left/input/{p}", f"/user/hand/right/input/{p}"]
binds = [("aim_pose", both("aim/pose")), ("grip_pose", both("grip/pose")), ("default_pose", both("aim/pose")),
         ("haptic", ["/user/hand/left/output/haptic", "/user/hand/right/output/haptic"]),
         ("trigger", both("trigger/value")), ("trigger_click", both("trigger/click")), ("trigger_touch", both("trigger/touch")),
         ("grip", both("squeeze/value")),
         ("primary", both("thumbstick")), ("primary_click", both("thumbstick/click")), ("primary_touch", both("thumbstick/touch")),
         ("bumper", both("bumper/click")),
         ("ax_button", ["/user/hand/right/input/a/click"]), ("by_button", ["/user/hand/right/input/b/click"]),
         ("x_button", ["/user/hand/right/input/x/click"]), ("y_button", ["/user/hand/right/input/y/click"]),
         ("menu_button", ["/user/hand/right/input/menu/click"]), ("view_button", ["/user/hand/left/input/view/click"]),
         ("dpad_up", ["/user/hand/left/input/dpad_up/click"]), ("dpad_down", ["/user/hand/left/input/dpad_down/click"]),
         ("dpad_left", ["/user/hand/left/input/dpad_left/click"]), ("dpad_right", ["/user/hand/left/input/dpad_right/click"])]
ids = []
for k, (a, paths) in enumerate(binds):
    for j, p in enumerate(paths):
        bid = f"OpenXRIPBinding_fr{k:02d}{j}"; ids.append(bid)
        blocks.append(f'[sub_resource type="OpenXRIPBinding" id="{bid}"]\naction = SubResource("{acts[a]}")\nbinding_path = "{p}"\n')
blocks.append('[sub_resource type="OpenXRInteractionProfile" id="OpenXRInteractionProfile_frame"]\n'
              'interaction_profile_path = "/interaction_profiles/valve/frame_controller_valve"\n'
              'bindings = [' + ", ".join(f'SubResource("{b}")' for b in ids) + ']\n')
i = t.index("\n[resource]")
t = t[:i] + "\n" + "\n".join(blocks) + t[i:]
t = re.sub(r'(interaction_profiles = \[)', r'\1SubResource("OpenXRInteractionProfile_frame"), ', t)
# action subresources must precede their use: move new action blocks before the action set
first_set = t.index('[sub_resource type="OpenXRActionSet"')
act_blocks = [b for b in blocks if 'type="OpenXRAction"' in b]
for b in act_blocks:
    t = t.replace(b, "", 1)
t = t[:first_set] + "\n".join(act_blocks) + "\n" + t[first_set:]
t = re.sub(r'load_steps=\d+ ', '', t)
open(fn, "w").write(t)
print(f"added {len(new_actions)} actions, {len(ids)} bindings")
