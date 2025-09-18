#!/usr/bin/env python3
"""
city_fragments_grid.py  —  fixed orientation & robust file loading

- Builds a rows×cols checkerboard of 1 m tiles (top of tiles is z=0 in the working frame).
- Loads every .glb/.gltf (case-insensitive) under --src recursively and flattens scene transforms.
- Converts each piece to Z-up for fitting, scales, anchors base/centroid to z=0, yaw, translate to tile center.
- Before export, rotates the entire scene to --export-up (default 'y') so it looks correct in Y-up viewers.

Examples (exactly 42 tiles):
  # Use your GLBs/GTLFs, uniform scale in all axes
  python city_fragments_grid.py --src "PATH_TO_YOUR_FILES" --out out/city.glb \
    --rows 6 --cols 7 --scale-mode uniform --no-keep-z --center-mode base --src-up y --export-up y
"""

import argparse, math, random, sys
from pathlib import Path
import numpy as np
import trimesh

# ---------------------------- grid visuals ----------------------------

def checker_rgba(i, j):
    return [220, 220, 225, 255] if ((i + j) % 2 == 0) else [180, 180, 190, 255]

def make_grid_scene(rows: int, cols: int, tile: float = 1.0, gap: float = 0.02,
                    thickness: float = 0.02, checker: bool = True) -> trimesh.Scene:
    """
    Create a grid of thin boxes whose *top faces* lie at z=0 (Z-up working frame).
    """
    scene = trimesh.Scene()
    face = max(1e-6, tile - gap)
    for r in range(rows):
        for c in range(cols):
            box = trimesh.creation.box(extents=(face, face, thickness))
            T = np.eye(4)
            T[:3, 3] = [c + 0.5, r + 0.5, -thickness * 0.5]  # center at z=-t/2, top at z=0
            box.apply_transform(T)
            if checker:
                rgba = np.array(checker_rgba(r, c), dtype=np.uint8)
                box.visual.face_colors = np.tile(rgba, (len(box.faces), 1))
            scene.add_geometry(box, node_name=f"tile_{r}_{c}")
    return scene

# ---------------------------- loading ----------------------------

def flatten_scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Robustly flatten a trimesh.Scene into a single Trimesh with transforms applied.
    Works across trimesh versions where nodes_geometry can be dict or list.
    """
    # Best path: built-in flattener
    try:
        return scene.to_mesh()
    except Exception:
        pass  # fall through to manual bake

    parts = []

    # Try dict form: {node_name: geometry_name}
    ng = getattr(scene.graph, "nodes_geometry", None)
    if isinstance(ng, dict):
        iterable = list(ng.items())
    elif isinstance(ng, (list, tuple)):
        # Some versions expose a list of (node_name, geometry_name) pairs,
        # or a list of node names; normalize to pairs when we can.
        pair_like = [(a, b) for a, b in ng if isinstance(ng[0], (list, tuple)) and len(ng[0]) >= 2] if ng else []
        if pair_like:
            iterable = pair_like
        else:
            iterable = []
    else:
        iterable = []

    if iterable:
        for node_name, geom_name in iterable:
            if geom_name not in scene.geometry:
                continue
            g = scene.geometry[geom_name].copy()
            try:
                T = scene.graph.get(node_name).matrix
            except Exception:
                T = np.eye(4)
            g.apply_transform(T)
            parts.append(g)
    else:
        # Last resort: walk all nodes, grab any that reference geometry
        try:
            for node_name in scene.graph.nodes:
                try:
                    geom_name = scene.graph[node_name].geometry  # may exist on some versions
                except Exception:
                    geom_name = None
                if geom_name and geom_name in scene.geometry:
                    g = scene.geometry[geom_name].copy()
                    try:
                        T = scene.graph.get(node_name).matrix
                    except Exception:
                        T = np.eye(4)
                    g.apply_transform(T)
                    parts.append(g)
        except Exception:
            # Absolute fallback: concatenate untransformed geometry
            parts = [g.copy() for g in scene.geometry.values()]

    return trimesh.util.concatenate(parts)

def load_fragment(path: Path) -> trimesh.Trimesh:
    """
    Load .glb/.gltf and return a single baked Trimesh.
    """
    obj = trimesh.load(path, force='scene')
    if isinstance(obj, trimesh.Scene):
        return flatten_scene_to_mesh(obj)
    if not isinstance(obj, trimesh.Trimesh):
        return obj.as_trimesh()
    return obj


def collect_paths(src: Path):
    exts = {'.glb', '.gltf'}
    return [p for p in src.rglob('*') if p.suffix.lower() in exts]

# ---------------------------- transforms ----------------------------

def rot_to_z_up(src_up: str, mins0=None, maxs0=None) -> np.ndarray:
    """
    Rotate *source* up-axis to Z-up (working frame).
    src_up: 'y' (glTF default), 'z', 'x', or 'auto'
    """
    if src_up == 'z':
        return np.eye(4)
    if src_up == 'x':
        return trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])  # X->Z
    if src_up == 'auto':
        if mins0 is not None and maxs0 is not None:
            size0 = maxs0 - mins0
            return trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0]) \
                   if size0[1] > size0[2] * 1.5 else np.eye(4)
        return np.eye(4)
    # default 'y' → rotate +90° about X so Y becomes Z
    return trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])

def world_to_export_up(export_up: str) -> np.ndarray:
    """
    Rotate the whole scene from Z-up (working frame) to requested export up-axis.
    """
    if export_up == 'z':
        return np.eye(4)
    if export_up == 'y':
        return trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])  # Z->Y
    if export_up == 'x':
        return trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])   # Z->X
    return np.eye(4)

def fit_transform_for_tile(
    mesh: trimesh.Trimesh,
    tile_center_xy,
    margin: float,
    scale_mode: str,
    keep_z: bool,
    center_mode: str,
    yaw_mode: str,
    src_up: str
) -> np.ndarray:
    """
    Compose a transform that orients to Z-up, centers, anchors, scales, yaws, then translates.
    """
    mins0, maxs0 = mesh.bounds
    R_up = rot_to_z_up(src_up, mins0, maxs0)

    work = mesh.copy()
    work.apply_transform(R_up)

    mins, maxs = work.bounds
    size = np.maximum(maxs - mins, 1e-12)

    avail = max(1e-6, 1.0 - 2.0 * margin)
    if scale_mode == "xy":
        sx = avail / size[0]
        sy = avail / size[1]
    else:
        s = avail / max(size[0], size[1])
        sx = sy = s

    sz = 1.0 if keep_z else min(sx, sy)

    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    z_anchor = mins[2] if center_mode == "base" else 0.5 * (mins[2] + maxs[2])

    T_to_origin = trimesh.transformations.translation_matrix([-cx, -cy, -z_anchor])
    S = np.diag([sx, sy, sz, 1.0])

    yaw_deg = 0.0 if yaw_mode == "none" else random.choice([0.0, 90.0, 180.0, 270.0])
    Rz = trimesh.transformations.rotation_matrix(np.deg2rad(yaw_deg), [0, 0, 1])

    T_to_tile = trimesh.transformations.translation_matrix([tile_center_xy[0], tile_center_xy[1], 0.0])

    return T_to_tile @ Rz @ S @ T_to_origin @ R_up

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Pack GLB/GTLF fragments onto a tiled board.")
    ap.add_argument("--src", type=str, required=False, default=".", help="Folder with .glb/.gltf (recursive)")
    ap.add_argument("--out", type=str, required=True, help="Output .glb")
    ap.add_argument("--rows", type=int, default=0, help="Rows; 0 = auto")
    ap.add_argument("--cols", type=int, default=0, help="Cols; 0 = auto")
    ap.add_argument("--margin", type=float, default=0.05, help="Margin inside each 1×1 tile")
    ap.add_argument("--gap", type=float, default=0.02, help="Gap between tiles")
    ap.add_argument("--thickness", type=float, default=0.02, help="Tile thickness")
    ap.add_argument("--scale-mode", choices=["uniform", "xy"], default="uniform", help="Uniform or per-axis XY")
    ap.add_argument("--keep-z", dest="keep_z", action="store_true", default=True, help="Keep Z scale (default)")
    ap.add_argument("--no-keep-z", dest="keep_z", action="store_false", help="Allow Z to scale with XY (uniform 3D)")
    ap.add_argument("--center-mode", choices=["base", "centroid"], default="base", help="Anchor base on plane or centroid")
    ap.add_argument("--yaw", dest="yaw_mode", choices=["none", "random"], default="none", help="Random 0/90/180/270 yaw per piece")
    ap.add_argument("--src-up", choices=["y", "z", "x", "auto"], default="y", help="Input up-axis (default y)")
    ap.add_argument("--export-up", choices=["y", "z", "x"], default="y", help="Scene up-axis in the exported GLB")
    ap.add_argument("--max", type=int, default=0, help="Limit number of files")
    ap.add_argument("--demo", type=int, default=0, help="Generate N synthetic shapes (ignores --src)")
    ap.add_argument("--no-grid", action="store_true", help="Do not create the floor grid")
    args = ap.parse_args()

    random.seed(0xC1F1)

    # Collect inputs
    pieces = []
    if args.demo > 0:
        for _ in range(args.demo):
            h = 0.5 + random.random() * 2.5
            b = 0.4 + random.random() * 0.8
            m = trimesh.creation.box(extents=[b, b*(0.6 + random.random()*0.8), h])
            if random.random() < 0.4:
                cone = trimesh.creation.cone(radius=0.2*b, height=0.3*h, sections=24)
                cone.apply_transform(trimesh.transformations.translation_matrix([0, 0, h/2 + 0.15*h]))
                m = trimesh.util.concatenate([m, cone])
            pieces.append(m)
        print(f"[INFO] DEMO mode: generated {len(pieces)} meshes.")
    else:
        src = Path(args.src)
        if not src.exists():
            sys.exit(f"[FATAL] --src not found: {src}")
        paths = collect_paths(src)
        if not paths:
            sys.exit(f"[FATAL] No .glb/.gltf found under: {src}")
        pieces = paths
        print(f"[INFO] found {len(paths)} files under {src}")
        for p in paths[:8]:
            print(f"       {p}")

    if args.max > 0:
        pieces = pieces[:args.max]

    N = len(pieces)
    cols = args.cols if args.cols > 0 else int(math.ceil(math.sqrt(N)))
    rows = args.rows if args.rows > 0 else int(math.ceil(N / max(cols, 1)))

    print(f"[INFO] placing {N} items on a {rows}×{cols} grid")

    scene = trimesh.Scene() if args.no_grid else make_grid_scene(rows, cols, 1.0, args.gap, args.thickness, True)

    r = c = 0
    for i in range(N):
        mesh = pieces[i].copy() if args.demo > 0 else load_fragment(Path(pieces[i]))
        if c >= cols:
            r += 1
            c = 0
        if r >= rows:
            print(f"[WARN] grid full; skipping index {i}")
            break

        tile_center = (c + 0.5, r + 0.5)
        M = fit_transform_for_tile(
            mesh, tile_center, args.margin, args.scale_mode, args.keep_z,
            args.center_mode, args.yaw_mode, args.src_up
        )
        mesh.apply_transform(M)

        if mesh.visual.kind != "face":
            mesh.visual.face_colors = [200, 210, 220, 255]

        scene.add_geometry(mesh, node_name=f"frag_{i}")
        c += 1

    # Convert from our Z-up working frame to requested export up-axis
    scene.apply_transform(world_to_export_up(args.export_up))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out, file_type='glb')
    print(f"[DONE] exported -> {out.resolve()}")
    print(f"[INFO] export-up: {args.export_up}   src-up: {args.src_up}   scale-mode: {args.scale_mode}   keep-z: {args.keep_z}")

if __name__ == "__main__":
    main()