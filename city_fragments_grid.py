#!/usr/bin/env python3
"""
city_fragments_grid.py

Builds a tiled "city board" and packs GLB fragments—one per 1x1 meter tile.
- Generates a plane subdivided into 1m x 1m squares (tiles), optionally checkered.
- Loads every *.glb in a folder (recursively), one per tile.
- Scales each fragment in X and Y to fit the 1x1 footprint (with margin), 
  lifts it so its base sits on z=0, and places it at the tile center.
- Exports a single GLB with the plane and all placed fragments.

Requires: numpy, trimesh
    pip install numpy trimesh

Usage:
    python city_fragments_grid.py --src path/to/glbs --out output/city.glb
    # optional knobs
    --rows 0 --cols 0           # auto (square-ish) if 0
    --margin 0.05               # free border inside each tile (meters)
    --gap 0.02                  # spacing between tiles (meters)
    --thickness 0.02            # tile thickness (meters)
    --scale-mode uniform|xy     # uniform preserves aspect; xy scales X & Y independently
    --keep-z                    # keep Z scale (default True). Use --no-keep-z to scale Z too.
    --center-mode base|centroid # 'base' places minZ on plane; 'centroid' centers vertically
    --yaw random|none           # optional random 0/90/180/270 yaw per piece
    --max N                     # limit how many GLBs are placed (default: all)
    --demo N                    # (optional) create N simple synthetic shapes instead of loading GLBs

Notes:
- Coordinates are in meters. The plane spans [0..cols] x [0..rows] at z=0 (top face).
- The script is conservative with materials; complex PBR may be simplified by trimesh's exporter.
"""

import argparse, math, os, random, sys
from pathlib import Path
import numpy as np
import trimesh

# ---------------------------- helpers ----------------------------

def make_checker_colors(i, j):
    """Return RGBA colors for alternating tiles."""
    if (i + j) % 2 == 0:
        return [220, 220, 225, 255]
    else:
        return [180, 180, 190, 255]

def make_grid_plane(rows: int, cols: int, tile: float = 1.0, gap: float = 0.02, thickness: float = 0.02,
                    checker: bool = True) -> trimesh.Scene:
    """
    Create a grid of thin boxes (tiles). The top of each tile sits at z=0.
    Returns a trimesh.Scene with every tile added.
    """
    scene = trimesh.Scene()
    for r in range(rows):
        for c in range(cols):
            # tile size with gap (gap is the space between tiles)
            face = tile - gap
            if face <= 0:
                face = tile * 0.95
            box = trimesh.creation.box(extents=(face, face, thickness))
            # center box such that top face is at z=0
            T = np.eye(4)
            T[:3, 3] = [c + 0.5, r + 0.5, -thickness / 2.0]  # Reverted: grid was fine, only geometry was wrong
            box.apply_transform(T)
            # simple color
            if checker:
                rgba = make_checker_colors(r, c)
                box.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8), (len(box.faces), 1))
            scene.add_geometry(box, node_name=f"tile_{r}_{c}")
    return scene

def load_fragment_as_mesh(glb_path: Path) -> trimesh.Trimesh:
    """
    Load a GLB. If it's a scene, flatten to a single mesh with transforms applied.
    """
    obj = trimesh.load(glb_path, force='scene')
    if isinstance(obj, trimesh.Scene):
        try:
            mesh = obj.to_mesh()
        except Exception:
            # Fallback: concatenate transformed geometries
            meshes = []
            for n in obj.graph.nodes_geometry:
                geom_name = obj.graph.nodes_geometry[n]
                geom = obj.geometry[geom_name].copy()
                T = obj.graph.get(n)[0] if hasattr(obj.graph, 'get') else obj.graph[n].matrix
                geom.apply_transform(T)
                meshes.append(geom)
            mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = obj
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.as_trimesh()
    return mesh

def fit_transform_for_tile(mesh: trimesh.Trimesh, tile_center_xy, 
                           margin: float, scale_mode: str,
                           keep_z: bool, center_mode: str, yaw_mode: str) -> np.ndarray:
    """
    Compute a world transform that:
      1) recenters mesh to its XY center and lifts base (or centroid) to z=0,
      2) scales in X/Y to fit into [0, 1] with margin,
      3) (optional) scales Z,
      4) applies an optional yaw,
      5) moves it to the tile center.
    """
    bounds = mesh.bounds   # (2, 3): min, max
    mins, maxs = bounds
    size = maxs - mins
    # Keep original geometry size - no scaling
    sx = sy = sz = 1.0

    # Optional yaw (rotation around Z)
    yaw_deg = 0.0
    if yaw_mode == "random":
        yaw_deg = random.choice([0.0, 90.0, 180.0, 270.0])
    yaw = np.deg2rad(yaw_deg)
    Rz = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1.0])
    
    # Fix model orientation: rotate 90 degrees around X-axis so models sit properly on the grid
    Rx = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])

    # Where to place vertically
    z_shift = -mins[2] if center_mode == "base" else -((mins[2] + maxs[2]) * 0.5)

    # Recenter to mesh bottom-left corner, lift to plane, scale, yaw, translate to tile center
    # Use bottom-left corner instead of center for proper grid alignment
    cx = mins[0]  # Use min X (left edge)
    cy = mins[1]  # Use min Y (bottom edge)

    T_to_origin = trimesh.transformations.translation_matrix([-cx, -cy, z_shift])
    S = np.diag([sx, sy, sz, 1.0])
    T_to_tile = trimesh.transformations.translation_matrix([tile_center_xy[0], tile_center_xy[1], 0.0])  # Fixed: keep original grid coordinates

    # Final transform (order: move to origin -> scale -> rotate to sit on grid -> yaw -> move to tile)
    M = T_to_tile @ Rz @ Rx @ S @ T_to_origin
    return M

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Pack GLB fragments into a tiled plane to form a 'city of fragments'.")
    ap.add_argument("--src", type=str, required=False, default=".", help="output\meshes")
    ap.add_argument("--out", type=str, required=True, help="output\3d")
    ap.add_argument("--rows", type=int, default=0, help="Rows in the grid; 0 = auto")
    ap.add_argument("--cols", type=int, default=0, help="Cols in the grid; 0 = auto")
    ap.add_argument("--margin", type=float, default=0.05, help="Margin inside each tile (meters)")
    ap.add_argument("--gap", type=float, default=0.02, help="Gap between tiles (meters)")
    ap.add_argument("--thickness", type=float, default=0.02, help="Tile thickness (meters)")
    ap.add_argument("--scale-mode", choices=["uniform", "xy"], default="uniform", help="Scale uniformly or per-axis in X/Y")
    ap.add_argument("--keep-z", dest="keep_z", action="store_true", default=True, help="Keep Z scale (default)")
    ap.add_argument("--no-keep-z", dest="keep_z", action="store_false", help="Allow Z to scale with XY")
    ap.add_argument("--center-mode", choices=["base", "centroid"], default="base", help="Place base on plane or center around plane")
    ap.add_argument("--yaw", dest="yaw_mode", choices=["none", "random"], default="none", help="Optional random 90° yaw per piece")
    ap.add_argument("--max", type=int, default=0, help="Limit how many GLBs to place (0 = all)")
    ap.add_argument("--demo", type=int, default=0, help="Generate N synthetic shapes instead of loading GLBs")
    args = ap.parse_args()

    random.seed(0xC1F1)  # Fixed: 'T' is not valid hex, changed to 'F'

    # Collect GLB files (unless demo)
    glbs = []
    if args.demo > 0:
        # Create synthetic meshes: extruded random prisms + boxes
        for i in range(args.demo):
            h = 0.5 + random.random() * 2.5
            b = 0.4 + random.random() * 0.8
            # box footprint ~1m before scaling
            m = trimesh.creation.box(extents=[b, b * (0.6 + random.random()*0.8), h])
            # add some roof flair
            if random.random() < 0.5:
                cone = trimesh.creation.cone(radius=0.2*b, height=0.3*h, sections=32)
                cone.apply_transform(trimesh.transformations.translation_matrix([0, 0, h/2 + 0.15*h]))
                m = trimesh.util.concatenate([m, cone])
            glbs.append(m)
    else:
        src = Path(args.src)
        if not src.exists():
            sys.exit(f"[FATAL] --src not found: {src}")
        for p in sorted(src.rglob("*.glb")):
            glbs.append(p)
        if not glbs:
            sys.exit(f"[FATAL] No .glb files found under: {src}")

    if args.max > 0:
        glbs = glbs[:args.max]

    N = len(glbs)
    # Auto grid if needed (square-ish)
    cols = args.cols if args.cols > 0 else int(math.ceil(math.sqrt(N)))
    rows = args.rows if args.rows > 0 else int(math.ceil(N / cols))

    print(f"[INFO] placing {N} items onto a {rows}x{cols} grid")

    # Create the tiled plane scene
    scene = make_grid_plane(rows, cols, tile=1.0, gap=args.gap, thickness=args.thickness, checker=True)

    # Place each fragment based on actual size
    current_col = 0
    current_row = 0
    
    for idx in range(N):
        if args.demo > 0:
            mesh = glbs[idx].copy()
        else:
            mesh = load_fragment_as_mesh(Path(glbs[idx]))

        # Calculate actual footprint size in grid units
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        footprint_cols = max(1, int(math.ceil(size[0])))  # X dimension in grid units
        footprint_rows = max(1, int(math.ceil(size[1])))  # Y dimension in grid units
        
        # Check if we have enough space in current row
        if current_col + footprint_cols > cols:
            # Move to next row
            current_row += 1
            current_col = 0
            
        # Check if we have enough rows
        if current_row + footprint_rows > rows:
            print(f"[WARNING] Not enough grid space for fragment {idx}, skipping...")
            continue
            
        # Calculate position so the model's footprint aligns with grid tiles
        # Place the model so its bottom-left corner aligns with the grid tile
        center_x = current_col + footprint_cols / 2.0
        center_y = current_row + footprint_rows / 2.0
        tile_center = (center_x, center_y)

        # Compute transform for this tile (no scaling)
        M = fit_transform_for_tile(mesh, tile_center, 0, args.scale_mode, args.keep_z, args.center_mode, args.yaw_mode)
        mesh.apply_transform(M)
        
        # Give it a neutral material color if missing
        if mesh.visual.kind == "face":
            # keep as-is
            pass
        else:
            mesh.visual.face_colors = [200, 210, 220, 255]

        scene.add_geometry(mesh, node_name=f"frag_{idx}")
        
        # Move to next position
        current_col += footprint_cols

    # Export
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path, file_type='glb')
    print(f"[DONE] exported -> {out_path.resolve()}")

if __name__ == "__main__":
    main()