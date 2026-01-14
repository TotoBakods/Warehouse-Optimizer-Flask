import math
import random
import numpy as np
import time
import multiprocessing
import gc
from functools import partial
from database import get_exclusion_zones

import atexit

# Global shared memory for multiprocessing (kept for backward compatibility)
_pool_items_props = None
_pool_wh_dims = None
_pool_valid_z = None
_pool_allocation_zones = None
_pool_exclusion_zones = None

_global_pool = None

def cleanup_pool():
    global _global_pool
    if _global_pool:
        _global_pool.terminate()
        _global_pool.join()
        _global_pool = None

atexit.register(cleanup_pool)

def get_process_pool():
    """Returns a singleton multiprocessing pool."""
    global _global_pool
    if _global_pool is None:
        cpu_count = multiprocessing.cpu_count()
        process_count = min(cpu_count, 12)
        _global_pool = multiprocessing.Pool(processes=process_count)
    return _global_pool

def init_worker(*args):
    """Deprecated: Initialization handled via explicit args now."""
    pass


def calculate_z_for_item(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h, other_items_stackable=None, strict_stacking=True):
    """Calculate the lowest valid Z position for an item based on items below it."""
    if len(other_items_bbox) == 0:
        return 0.0
        
    # New item bounding box
    new_min_x = x - dim_x / 2
    new_max_x = x + dim_x / 2
    new_min_y = y - dim_y / 2
    new_max_y = y + dim_y / 2
    
    # Check XY plane overlaps (vectorized)
    overlaps_x = (new_min_x < other_items_bbox[:, 2]) & (new_max_x > other_items_bbox[:, 0])
    overlaps_y = (new_min_y < other_items_bbox[:, 3]) & (new_max_y > other_items_bbox[:, 1])
    overlaps = overlaps_x & overlaps_y
    
    if not np.any(overlaps):
        return 0.0
        
    # Get max Z top of overlapping items
    overlapping_z_tops = other_items_z[overlaps] + other_items_h[overlaps]

    # Reject stacking on non-stackable items
    if strict_stacking and other_items_stackable is not None:
        overlapping_stackables = other_items_stackable[overlaps]
        if np.any(overlapping_stackables == 0):
             return 1000000.0 # Effectively impossible
             
    max_z = np.max(overlapping_z_tops)
    
    # Stability check: ensure sufficient support area
    if max_z > 0:
        is_support = np.abs(overlapping_z_tops - max_z) < 0.01
        
        all_indices = np.arange(len(other_items_bbox))
        overlapping_indices = all_indices[overlaps]
        support_indices = overlapping_indices[is_support]
        
        # Calculate intersection area with supporting items
        sup_min_x = other_items_bbox[support_indices, 0]
        sup_min_y = other_items_bbox[support_indices, 1]
        sup_max_x = other_items_bbox[support_indices, 2]
        sup_max_y = other_items_bbox[support_indices, 3]
        
        inter_min_x = np.maximum(new_min_x, sup_min_x)
        inter_max_x = np.minimum(new_max_x, sup_max_x)
        inter_min_y = np.maximum(new_min_y, sup_min_y)
        inter_max_y = np.minimum(new_max_y, sup_max_y)
        
        w = np.maximum(0, inter_max_x - inter_min_x)
        h = np.maximum(0, inter_max_y - inter_min_y)
        area = w * h
        supported_area = np.sum(area)
        
        item_area = dim_x * dim_y
        if supported_area < (item_area * 0.2):  # 20% support threshold
            return max_z + 100000.0
            
    return max_z



def get_rotated_dims(l, w, h, rotation_code):
    """Returns (dx, dy, dz) based on rotation code 0-5."""
    code = int(rotation_code) % 6
    if code == 0: return l, w, h
    if code == 1: return w, l, h
    if code == 2: return l, h, w
    if code == 3: return h, l, w
    if code == 4: return w, h, l
    if code == 5: return h, w, l
    return l, w, h

def repair_solution_compact(solution, items_props=None, warehouse_dims=None, allocation_zones=None, layer_heights=None):
    """Repair solution by placing items in valid positions with gravity."""
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    
    # Defaults
    if layer_heights is None or len(layer_heights) == 0:
        layer_heights = [0.0]
    
    wh_len = warehouse_dims[0] if warehouse_dims else 100
    wh_wid = warehouse_dims[1] if warehouse_dims else 100
    wh_hgt = warehouse_dims[2] if warehouse_dims else 10
    
    num_items = len(solution)
    if num_items == 0: return solution

    # Sort indices:
    # Priority 1: Fragility (Ascending) - 0 (Robust) first, 1 (Fragile) last
    # Priority 2: Weight (Descending) - Heavy items first
    # Priority 3: Volume (Descending) - Large items first
    fragility = items_props[:, 8]
    weights = items_props[:, 6]
    volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]
    
    indices = np.arange(num_items)
    # Sort: fragile last, then heavy/large first
    sorted_indices = sorted(indices, key=lambda i: (fragility[i], -weights[i], -volumes[i], i))

    # Tracking placed items: (x, y, z, dx, dy, dz)
    placed_items = []
    
    # Use provided zones or default to full warehouse
    use_zones = allocation_zones if allocation_zones else [{'x1':0, 'y1':0, 'x2':wh_len, 'y2':wh_wid, 'z1':0, 'z2':wh_hgt}]

    for idx in sorted_indices:
        l, w, h = items_props[idx, 0:3]
        can_rotate = int(items_props[idx, 3])
        
        # Try flat rotations for stability
        rots = [0, 1] if can_rotate else [int(solution[idx, 3])]
        if can_rotate and items_props[idx, 4] == 1:
             pass  # Stick to flat rotations
        
        best_pos = None
        
        # Generate candidate positions
        candidates = set()
        
        # From zone corners
        for z in use_zones:
            candidates.add((z['x1'], z['y1']))
        
        # From placed items (adjacent positions)
        for (px, py, pz, pdx, pdy, pdz) in placed_items:
            candidates.add((px + pdx, py))  # Right
            candidates.add((px, py + pdy))  # Back
            candidates.add((px, py))        # On top
        
        # Filter valid candidates
        valid_candidates = []
        for (cx, cy) in candidates:
             if cx >= 0 and cy >= 0 and cx < wh_len and cy < wh_wid:
                 valid_candidates.append((cx, cy))
                 
        # Sort by proximity to optimizer's suggested position
        target_x = solution[idx, 0]
        target_y = solution[idx, 1]
        
        sorted_candidates = sorted(valid_candidates, key=lambda p: (
            (p[0] - target_x)**2 + (p[1] - target_y)**2,
            p[1], p[0]
        ))
        
        for rot in rots:
            dims = get_rotated_dims(l, w, h, rot)
            dx, dy, dz = dims
            
            for (cx, cy) in sorted_candidates:
                min_x, min_y = cx, cy
                max_x, max_y = cx + dx, cy + dy
                
                if max_x > wh_len + 0.001 or max_y > wh_wid + 0.001:
                    continue
                
                # Calculate gravity Z
                gravity_z = 0.0
                
                # Find highest item below this footprint
                for (px, py, pz, pdx, pdy, pdz) in placed_items:
                    if (max_x > px + 0.001 and min_x < px + pdx - 0.001 and
                        max_y > py + 0.001 and min_y < py + pdy - 0.001):
                        top_z = pz + pdz
                        if top_z > gravity_z:
                            gravity_z = top_z
                
                # Find valid Z in any suitable zone
                valid_z_found = False
                final_z = float('inf')
                
                for zne in use_zones:
                    # Check XY containment
                    if (min_x >= zne['x1'] - 0.01 and max_x <= zne['x2'] + 0.01 and 
                        min_y >= zne['y1'] - 0.01 and max_y <= zne['y2'] + 0.01):
                        
                        zone_floor = zne.get('z1', 0)
                        placement_z = max(gravity_z, zone_floor)
                        placement_top = placement_z + dz
                        zone_ceil = zne.get('z2', wh_hgt)
                        
                        # Check Z fits
                        if placement_top <= zone_ceil + 0.001:
                            if placement_z < final_z:
                                final_z = placement_z
                                valid_z_found = True
                    
                if valid_z_found:
                    # Calculate final score (Z, Y, X)
                    score = (final_z, min_y, min_x)
                    if best_pos is None or score < best_pos[7]:
                         center_x = min_x + dx/2
                         center_y = min_y + dy/2
                         best_pos = (center_x, center_y, final_z, rot, dx, dy, dz, score)
    
        # Apply placement
        if best_pos:
            b_x, b_y, b_z, b_rot, b_dx, b_dy, b_dz, _ = best_pos
        else:
            # Fallback: stack on top of everything
            b_z = 0
            if placed_items:
                max_top = max([p[2]+p[5] for p in placed_items])
                b_z = max_top
            
            b_rot = solution[idx, 3] if not can_rotate else 0
            dims = get_rotated_dims(l, w, h, b_rot)
            b_dx, b_dy, b_dz = dims
            b_x = dims[0]/2
            b_y = dims[1]/2
            
        solution[idx, 0] = b_x
        solution[idx, 1] = b_y
        solution[idx, 2] = b_z
        solution[idx, 3] = b_rot
        
        placed_items.append((b_x - b_dx/2, b_y - b_dy/2, b_z, b_dx, b_dy, b_dz))

    return solution


# Get valid Z positions for layers
def get_valid_z_positions(warehouse):
    if 'layer_heights' in warehouse and warehouse['layer_heights'] is not None:
        positions = set(warehouse['layer_heights'])
        positions.add(0.0)
        return sorted(list(positions))

    levels = warehouse.get('levels', 1)
    if levels <= 1:
        return [0.0]
    height = warehouse.get('height', 1)
    level_height = height / levels if levels > 0 else 0
    return [i * level_height for i in range(levels)]


# Standalone functions for multiprocessing

def create_random_solution_array(num_items, warehouse_dims=None, items_props=None, allocation_zones=None):
    """Create a random solution array with gravity-based placement."""
    # Use globals if running in worker
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    
    
    solution = np.zeros((num_items, 4), dtype=np.float32)
    wh_len, wh_wid, wh_hgt = warehouse_dims[:3]
    
    # Check for allocation zones
    has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
    
    # Track placed items for gravity
    placed_bboxes = np.zeros((num_items, 4), dtype=np.float32)
    placed_z = np.zeros(num_items, dtype=np.float32)
    placed_h = np.zeros(num_items, dtype=np.float32)
    
    for i in range(num_items):
        item_len = items_props[i, 0]
        item_wid = items_props[i, 1]
        item_hgt = items_props[i, 2]
        can_rotate = items_props[i, 3]
        
        # Retry for floor priority (try to find Z=0)
        best_x, best_y, best_z = 0, 0, float('inf')
        best_rotation = 0
        
        for attempt in range(50):
            # Randomize rotation
            rotation = 0
            if can_rotate and random.random() > 0.5:
                    rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            
            # Position selection logic
            valid_zones = []
            if has_allocation_zones:
                for zone in allocation_zones:
                    zone_width = zone['x2'] - zone['x1']
                    zone_depth = zone['y2'] - zone['y1']
                    if dim_x <= zone_width and dim_y <= zone_depth:
                        valid_zones.append(zone)
            
            zone_z1 = 0
            zone_z2 = wh_hgt
            if valid_zones:
                # Sort zones by Z (bottom first)
                valid_zones.sort(key=lambda z: z.get('z1', 0))
                
                # Select zone sequentially
                zone_idx = attempt % len(valid_zones)
                zone_idx = min(zone_idx, len(valid_zones) - 1)
                zone = valid_zones[zone_idx]
                
                zone_z1 = zone.get('z1', 0)
                zone_z2 = zone.get('z2', wh_hgt)
                
                min_x = zone['x1'] + dim_x / 2
                max_x = zone['x2'] - dim_x / 2
                min_y = zone['y1'] + dim_y / 2
                max_y = zone['y2'] - dim_y / 2
                
                if max_x < min_x: max_x = min_x = (zone['x1'] + zone['x2']) / 2
                if max_y < min_y: max_y = min_y = (zone['y1'] + zone['y2']) / 2
                
                # Dense packing: try corner first, then adjacent, then random
                if attempt < 5:
                    x = min_x
                    y = min_y
                elif attempt < 45 and i > 0:
                    # Place adjacent to existing item
                    rand_idx = random.randint(0, i-1)
                    ref_box = placed_bboxes[rand_idx]
                    if random.random() < 0.5:
                        x = ref_box[2] + dim_x / 2
                        y = ref_box[1] + dim_y / 2
                    else:
                        x = ref_box[0] + dim_x / 2
                        y = ref_box[3] + dim_y / 2
                    x += random.uniform(-1, 1)
                    y += random.uniform(-1, 1)
                    x = max(min_x, min(max_x, x))
                    y = max(min_y, min(max_y, y))
                else:
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    
            else:
                min_x = dim_x / 2
                max_x = wh_len - dim_x / 2
                min_y = dim_y / 2
                max_y = wh_wid - dim_y / 2
                
                if max_x < min_x: max_x = min_x
                if max_y < min_y: max_y = min_y
                
                # Dense packing (global)
                if attempt < 3:
                        x = min_x
                        y = min_y
                elif attempt < 6 and i > 0:
                        rand_idx = random.randint(0, i-1)
                        ref_box = placed_bboxes[rand_idx]
                        if random.random() < 0.5:
                            x = ref_box[2] + dim_x / 2
                            y = ref_box[1] + dim_y / 2
                        else:
                            x = ref_box[0] + dim_x / 2
                            y = ref_box[3] + dim_y / 2
                        x = max(min_x, min(max_x, x))
                        y = max(min_y, min(max_y, y))
                else:
                        x = round(random.uniform(min_x, max_x))
                        y = round(random.uniform(min_y, max_y))
            
            # Check Z immediately
            z = calculate_z_for_item(x, y, dim_x, dim_y, placed_bboxes[:i], placed_z[:i], placed_h[:i])
            
            # Enforce layer floor
            z = max(z, zone_z1)
            
            # Snap to next layer if exceeds ceiling
            if z + item_hgt > zone_z2 and zone_z2 < wh_hgt:
                z = zone_z2
            
            if z < best_z:
                best_x, best_y, best_z, best_rotation = x, y, z, rotation
            
            if z == zone_z1:
                break
        
        if best_z > 50000:
            best_z -= 100000.0
        
        x, y, z = best_x, best_y, best_z
        rotation = best_rotation
        
        # Recalculate dims for best rotation
        if int(rotation) % 180 == 0:
            dim_x, dim_y = item_len, item_wid
        else:
            dim_x, dim_y = item_wid, item_len
        
        # Store
        solution[i] = [x, y, z, rotation]
        
        # Update tracking arrays for next items
        placed_bboxes[i] = [x - dim_x/2, y - dim_y/2, x + dim_x/2, y + dim_y/2]
        placed_z[i] = z
        placed_h[i] = item_hgt
        
    return solution

def fitness_function_numpy(solution, items_props=None, warehouse_dims=None, weights=None, valid_z=None, exclusion_zones_arr=None):
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if valid_z is None: valid_z = _pool_valid_z
    if exclusion_zones_arr is None: exclusion_zones_arr = _pool_exclusion_zones
    
    # solution: (N, 4)
    # items_props: (N, 8) cols: len, wid, hgt, can_rot, stackable, access_freq, weight, category_id
    # exclusion_zones_arr: (K, 4) -> x1, y1, x2, y2
    
    # Ensure float32 for memory efficiency
    solution = solution.astype(np.float32, copy=False)
    # items_props is likely already float32 if we initialized it carefully, but let's assume it's ro (read-only)
    
    # Calculate Space Utilization
    # Calculate Space Utilization
    grouping = 0.0 # Initialize early to avoid NameError
    total_vol = np.sum(items_props[:, 0] * items_props[:, 1] * items_props[:, 2])
    wh_vol = warehouse_dims[0] * warehouse_dims[1] * warehouse_dims[2]
    space_util = total_vol / wh_vol if wh_vol > 0 else 0
    
    # Calculate Accessibility (Distance to door)
    door_x, door_y = 0, 0
    if len(warehouse_dims) >= 5:
        door_x, door_y = warehouse_dims[3], warehouse_dims[4]
        
    dists = np.sqrt((solution[:, 0] - door_x)**2 + (solution[:, 1] - door_y)**2)
    # Avoid div by zero
    access_scores = 1.0 / (1.0 + dists)
    
    freqs = items_props[:, 5]
    if np.sum(freqs) > 1e-9:
        accessibility = np.average(access_scores, weights=freqs)
    else:
        accessibility = np.mean(access_scores)
    
    # Stability
    on_valid_z = np.zeros(len(solution), dtype=bool)
    if valid_z:
        for z_pos in valid_z:
            on_valid_z |= (np.abs(solution[:, 2] - z_pos) < 0.001)
    stability = np.mean(on_valid_z)
    
    # Exclusion Zones
    zone_penalty = 0
    if exclusion_zones_arr is not None and len(exclusion_zones_arr) > 0:
        x = solution[:, 0:1] # (N, 1)
        y = solution[:, 1:2] # (N, 1)
        
        z_x1 = exclusion_zones_arr[:, 0] # (K,)
        z_y1 = exclusion_zones_arr[:, 1]
        z_x2 = exclusion_zones_arr[:, 2]
        z_y2 = exclusion_zones_arr[:, 3]
        
        # Better: AABB overlap
        # Item dims (approximation with non-rotated len/wid for speed or use max dim)
        radii = np.maximum(items_props[:, 0], items_props[:, 1]) / 2.0
        radii = radii.reshape(-1, 1)
        
        # Zone centers/dims
        z_cx = (z_x1 + z_x2) / 2
        z_cy = (z_y1 + z_y2) / 2
        z_hw = (z_x2 - z_x1) / 2
        z_hh = (z_y2 - z_y1) / 2
        
        # Distance from center to center per axis
        dx = np.abs(x - z_cx)
        dy = np.abs(y - z_cy)
        
        # Vectorized AABB with rotation ignored (using max dim covers worst case)
        collision_x = dx < (radii + z_hw)
        collision_y = dy < (radii + z_hh)
        collisions = collision_x & collision_y
        
        zone_penalty = np.sum(collisions) / len(solution) # Fraction of items colliding
        
    # --- Item-Item Overlap ---
    # Optimized to use batches to avoid O(N^2) memory usage for large N.
    n = len(solution)
    overlap_count = 0
    if n > 0:
        # Extract centers (n, 1)
        x = solution[:, 0]
        y = solution[:, 1]
        z = solution[:, 2]  # Bottom
        h = items_props[:, 2]  # Heights
        
        # Calculate actual dimensions based on rotation
        rots = solution[:, 3].astype(int)
        l = items_props[:, 0]
        w = items_props[:, 1]
        orig_h = items_props[:, 2] # Need original height to swap
        
        # We need to vectorizely apply get_rotated_dims?
        # get_rotated_dims is not vectorized.
        # But we can simulate it with numpy usage.
        
        # Codes 0-5
        # 0: L, W, H
        # 1: W, L, H
        # 2: L, H, W
        # 3: H, L, W
        # 4: W, H, L
        # 5: H, W, L
        
        current_len = np.zeros(n, dtype=np.float32)
        current_wid = np.zeros(n, dtype=np.float32)
        current_hgt = np.zeros(n, dtype=np.float32)
        
        rot_mod = rots % 6
        
        # Case 0
        mask = (rot_mod == 0)
        current_len[mask] = l[mask]
        current_wid[mask] = w[mask]
        current_hgt[mask] = orig_h[mask]
        
        # Case 1
        mask = (rot_mod == 1)
        current_len[mask] = w[mask]
        current_wid[mask] = l[mask]
        current_hgt[mask] = orig_h[mask]
        
        # Case 2
        mask = (rot_mod == 2)
        current_len[mask] = l[mask]
        current_wid[mask] = orig_h[mask]
        current_hgt[mask] = w[mask]
        
        # Case 3
        mask = (rot_mod == 3)
        current_len[mask] = orig_h[mask]
        current_wid[mask] = l[mask]
        current_hgt[mask] = w[mask]
        
        # Case 4
        mask = (rot_mod == 4)
        current_len[mask] = w[mask]
        current_wid[mask] = orig_h[mask]
        current_hgt[mask] = l[mask]
        
        # Case 5
        mask = (rot_mod == 5)
        current_len[mask] = orig_h[mask]
        current_wid[mask] = w[mask]
        current_hgt[mask] = l[mask]
        
        # Half-dims
        hw = current_len / 2
        hh = current_wid / 2
        
        # Z intervals (bottom + rotated height)
        z1 = z
        z2 = z + current_hgt
        
        # Reduced Batch Size for Memory Safety
        BATCH_SIZE = 512 
        
        for i_start in range(0, n, BATCH_SIZE):
            i_end = min(i_start + BATCH_SIZE, n)
            
            # Batch slices
            x_batch = x[i_start:i_end].reshape(-1, 1)  # (B, 1)
            y_batch = y[i_start:i_end].reshape(-1, 1)
            z1_batch = z1[i_start:i_end].reshape(-1, 1)
            z2_batch = z2[i_start:i_end].reshape(-1, 1)
            hw_batch = hw[i_start:i_end].reshape(-1, 1)
            hh_batch = hh[i_start:i_end].reshape(-1, 1)
            
            # Inner loop batching to keep memory low
            for j_start in range(0, n, BATCH_SIZE):
                j_end = min(j_start + BATCH_SIZE, n)
                
                # Check bounds to skip duplicate work if we wanted to triangularize, 
                # but calculating full matrix blockwise is simpler to vectorize.
                
                x_other = x[j_start:j_end].reshape(1, -1)  # (1, B2)
                y_other = y[j_start:j_end].reshape(1, -1)
                z1_other = z1[j_start:j_end].reshape(1, -1)
                z2_other = z2[j_start:j_end].reshape(1, -1)
                hw_other = hw[j_start:j_end].reshape(1, -1)
                hh_other = hh[j_start:j_end].reshape(1, -1)
                
                # Overlap checks
                # X overlap: |x1 - x2| < hw1 + hw2
                dx = np.abs(x_batch - x_other)
                overlap_x = dx < (hw_batch + hw_other - 0.01) # 1cm tolerance? No, stricter.
                
                # Y overlap
                dy = np.abs(y_batch - y_other)
                overlap_y = dy < (hh_batch + hh_other - 0.01)
                
                # Z overlap
                # Interval overlap: not (end1 <= start2 or start1 >= end2)
                # Strict < inequality implies 0 thickness overlap is ignored, which is good.
                overlap_z = (z2_batch > (z1_other + 0.01)) & (z1_batch < (z2_other - 0.01))
                
                # Combined
                overlaps = overlap_x & overlap_y & overlap_z
                
                overlap_count += np.sum(overlaps)
        
        # Remove self-overlaps (diagonal was counted once per item)
        # Each item overlaps with itself in the logic above.
        overlap_count -= n
        
        # Divide by 2 because A-B and B-A are counted
        overlap_count /= 2.0
        
        # Draconian Penalty
        if overlap_count > 0:
             overlap_penalty = overlap_count * 10000.0 # Extreme penalty
        else:
             overlap_penalty = 0.0
        
    # --- Stackability Enforcement ---
    # Check if items are stacked on non-stackable items
    stackability_penalty = 0
    
    # Optimized Vectorized Stackability Check
    # n is already defined
    if n > 1:
        # Reuse variables extracted earlier
        x = solution[:, 0]
        y = solution[:, 1]
        z = solution[:, 2]
        h = items_props[:, 2]  # heights
        stackable = items_props[:, 4]  # stackable flags
        
        # Get item footprint dimensions (accounting for rotation)
        # Get item footprint dimensions (accounting for rotation)
        rots = solution[:, 3].astype(int)
        l = items_props[:, 0]
        w = items_props[:, 1]
        orig_h = items_props[:, 2]
        
        # 6-Axis Dimension Logic (Vectorized)
        current_len = np.zeros(n, dtype=np.float32)
        current_wid = np.zeros(n, dtype=np.float32)
        current_hgt = np.zeros(n, dtype=np.float32) # We need this for z_tops!
        
        rot_mod = rots % 6
        
        # Case 0 (L, W, H)
        mask = (rot_mod == 0)
        current_len[mask] = l[mask]; current_wid[mask] = w[mask]; current_hgt[mask] = orig_h[mask]
        # Case 1 (W, L, H)
        mask = (rot_mod == 1)
        current_len[mask] = w[mask]; current_wid[mask] = l[mask]; current_hgt[mask] = orig_h[mask]
        # Case 2 (L, H, W)
        mask = (rot_mod == 2)
        current_len[mask] = l[mask]; current_wid[mask] = orig_h[mask]; current_hgt[mask] = w[mask]
        # Case 3 (H, L, W)
        mask = (rot_mod == 3)
        current_len[mask] = orig_h[mask]; current_wid[mask] = l[mask]; current_hgt[mask] = w[mask]
        # Case 4 (W, H, L)
        mask = (rot_mod == 4)
        current_len[mask] = w[mask]; current_wid[mask] = orig_h[mask]; current_hgt[mask] = l[mask]
        # Case 5 (H, W, L)
        mask = (rot_mod == 5)
        current_len[mask] = orig_h[mask]; current_wid[mask] = w[mask]; current_hgt[mask] = l[mask]

        hw = current_len / 2
        hh = current_wid / 2
        
        # Z-tops MUST use rotated height
        z_tops = z + current_hgt
        
        # We need to find pairs (i, j) where i is resting on j.
        # Resting condition: abs(z[i] - z_top[j]) < 0.1
        # AND Footprint Overlap
        
        violations = 0
        BATCH_SIZE_STACK = 128 # Small batch for safety
        
        for i_start in range(0, n, BATCH_SIZE_STACK):
            i_end = min(i_start + BATCH_SIZE_STACK, n)
            
            # Batch I data (Potential Top Items)
            z_i = z[i_start:i_end].reshape(-1, 1) # (B, 1)
            x_i = x[i_start:i_end].reshape(-1, 1)
            y_i = y[i_start:i_end].reshape(-1, 1)
            hw_i = hw[i_start:i_end].reshape(-1, 1)
            hh_i = hh[i_start:i_end].reshape(-1, 1)
            
            # Filter: Only check items that are NOT on the ground
            # effective_mask = (z_i > 0.01).flatten()
            # If we want to optimize further we could skip ground items, but vectorization is fast enough.
            
            for j_start in range(0, n, BATCH_SIZE_STACK):
                j_end = min(j_start + BATCH_SIZE_STACK, n)
                
                # Batch J data (Potential Support Items)
                z_j_top = z_tops[j_start:j_end].reshape(1, -1) # (1, B2)
                
                # Z-Check: Is i resting on j?
                # resting = abs(z_i - z_j_top) < 0.1
                resting = np.abs(z_i - z_j_top) < 0.1
                
                if not np.any(resting):
                    continue
                    
                # XY Overlap Check for resting pairs
                x_j = x[j_start:j_end].reshape(1, -1)
                y_j = y[j_start:j_end].reshape(1, -1)
                hw_j = hw[j_start:j_end].reshape(1, -1)
                hh_j = hh[j_start:j_end].reshape(1, -1)
                
                dx = np.abs(x_i - x_j)
                dy = np.abs(y_i - y_j)
                
                # Overlap Threshold (50% rule mostly... logic was: < (hw1+hw2)*0.5)
                # Wait, original logic: overlap_threshold_x = (hw[i] + hw[j]) * 0.5
                # This means centers must be VERY close.
                # Actually (hw[i] + hw[j]) is the touching distance. * 0.5 means they must overlap by 50%?
                # Yes.
                
                thresh_x = (hw_i + hw_j) * 0.5
                thresh_y = (hh_i + hh_j) * 0.5
                
                xy_overlap = (dx < thresh_x) & (dy < thresh_y)
                
                # Valid Support Relation
                is_supported = resting & xy_overlap
                
                # Self-support is impossible due to z vs z+h check (unless h=0, but valid check prevents self-loop effectively)
                # But to be safe, if i==j, z_i cannot equal z_j + h_j unless h_j=0.
                
                if np.any(is_supported):
                     # Check if supporter J is stackable
                     stackable_j = stackable[j_start:j_end].reshape(1, -1) # (1, B2)
                     
                     # Identify bad supports: Supported by item with stackable=0
                     # mask: is_supported AND (stackable_j == 0)
                     bad_support = is_supported & (stackable_j < 0.5)
                     
                     # Count unique ITEMS 'i' that have at least one bad support
                     # Reduce along J axis: does item i have ANY bad support in this batch?
                     has_bad_support = np.any(bad_support, axis=1) # (B,)
                     
                     violations += np.sum(has_bad_support)

        stackability_penalty = violations / n if n > 0 else 0
    
    # --- Grouping Metric ---
    grouping = 0.0
    n_items = len(solution)
    if n_items > 0:
        cats = items_props[:, 7] # Category hash
        unique_cats = np.unique(cats)
        
        total_dist_sum = 0
        count = 0
        
        x = solution[:, 0]
        y = solution[:, 1]
        
        for cat in unique_cats:
            # Mask for this category
            mask = (cats == cat)
            if np.sum(mask) <= 1:
                continue # Single item already grouped with itself
                
            # Centroid
            c_x = np.mean(x[mask])
            c_y = np.mean(y[mask])
            
            # Distances to centroid
            dists = np.sqrt((x[mask] - c_x)**2 + (y[mask] - c_y)**2)
            
            total_dist_sum += np.sum(dists)
            count += np.sum(mask)
        
        if count > 0:
            avg_dist = total_dist_sum / count
            grouping = 1.0 / (1.0 + avg_dist * 0.1)
        else:
            grouping = 1.0 # Perfect grouping if all singles or empty
    else:
            grouping = 0

    
    total_weight = sum(weights.values())
    if total_weight <= 1e-9:
        norm_weights = weights # Avoid division by zero
    else:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Penalize fitness for zone violations
    fitness = (norm_weights.get('space', 0) * space_util +
                norm_weights.get('accessibility', 0) * accessibility +
                norm_weights.get('stability', 0) * stability +
                norm_weights.get('grouping', 0) * grouping)
    
    if zone_penalty > 0:
        fitness *= (1.0 - zone_penalty) # Reduce fitness by % colliding

    if overlap_penalty > 0:
        fitness *= max(0, (1.0 - overlap_penalty * 5.0)) # 20% overlap = 0 fitness

    # Apply stackability penalty - items on non-stackable items
    if stackability_penalty > 0:
        fitness *= max(0, (1.0 - stackability_penalty * 3.0))

    if fitness <= 1e-6:
        # Debug only randomly to save IO
        if random.random() < 0.001:
            with open('thread_debug.log', 'a') as f:
                f.write(f"Zero Fit: Overlap={overlap_penalty:.4f}, Zone={zone_penalty:.4f}, Stack={stackability_penalty:.4f}\n")

    # Prefer lower Z (floor usage) if possible - reducing fitness slightly as average Z increases
    avg_z = np.mean(solution[:, 2])
    wh_hgt_val = warehouse_dims[2]
    if wh_hgt_val > 0:
        fitness *= (1.0 - (avg_z / wh_hgt_val) * 0.15) 

    return fitness, space_util, accessibility, stability, grouping

def create_and_repair(num_items, warehouse_dims, items_props, allocation_zones, valid_z):
    sol = create_random_solution_array(num_items, warehouse_dims, items_props, allocation_zones)
    sol = repair_solution_compact(sol, items_props, warehouse_dims, allocation_zones, valid_z)
    return sol

def crossover_numpy(parent1, parent2, crossover_rate):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
        
    if len(parent1) < 2:
        return parent1.copy(), parent2.copy()

    # Single point crossover along the item axis
    point = random.randint(1, len(parent1) - 1)
    child1 = np.vstack((parent1[:point], parent2[point:]))
    child2 = np.vstack((parent2[:point], parent1[point:]))
    return child1, child2

def mutation_numpy(solution, warehouse_dims=None, items_props=None, valid_z=None, mutation_rate=0.01, allocation_zones=None):
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if valid_z is None: valid_z = _pool_valid_z
    if allocation_zones is None: allocation_zones = _pool_allocation_zones

    # Randomly mutate 1 item
    if random.random() < mutation_rate:
        idx = random.randint(0, len(solution) - 1)
        
        # Re-randomize this item
        item_len = items_props[idx, 0]
        item_wid = items_props[idx, 1]
        item_hgt = items_props[idx, 2]
        can_rotate = items_props[idx, 3]
        
        wh_len, wh_wid, wh_hgt = warehouse_dims[0], warehouse_dims[1], warehouse_dims[2]
        
        # --- Gravity Calculation Prep ---
        # Pre-calculate other items' bboxes ONCE
        num_items = len(solution)
        item_to_mutate = idx
        mask = np.arange(num_items) != item_to_mutate
        other_solution = solution[mask]
        other_props = items_props[mask]
        
        other_x = other_solution[:, 0]
        other_y = other_solution[:, 1]
        other_z = other_solution[:, 2]
        other_rot = other_solution[:, 3]
        other_l = other_props[:, 0]
        other_w = other_props[:, 1]
        other_h = other_props[:, 2]
        
        swap_mask = (other_rot.astype(int) % 180 != 0)
        cur_l = np.where(swap_mask, other_w, other_l)
        cur_w = np.where(swap_mask, other_l, other_w)
        
        other_bbox = np.zeros((len(other_solution), 4))
        other_bbox[:, 0] = other_x - cur_l / 2
        other_bbox[:, 1] = other_y - cur_w / 2
        other_bbox[:, 2] = other_x + cur_l / 2
        other_bbox[:, 3] = other_y + cur_w / 2
        
        # Retry loop for floor priority
        best_x, best_y, best_z = 0, 0, float('inf')
        best_rotation = 0
        
        for attempt in range(50):
            # Dynamic Rotation: Randomize orientation per attempt
            rotation = solution[idx, 3] # Default to current
            if can_rotate and random.random() > 0.5:
                 rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            # Check if we should use allocation zones
            has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
            
            zone_z1 = 0
            zone_z2 = wh_hgt
            if has_allocation_zones:
                # Find zones that can fit this item
                valid_zones = []
                for zone in allocation_zones:
                    zone_width = zone['x2'] - zone['x1']
                    zone_depth = zone['y2'] - zone['y1']
                    zone_z1_val = zone.get('z1', 0)
                    zone_z2 = zone.get('z2', wh_hgt)
                    
                    if dim_x <= zone_width and dim_y <= zone_depth and item_hgt <= (zone_z2 - zone_z1_val):
                        valid_zones.append(zone)
                
                if valid_zones:
                    # Sort zones by Z height (Bottom Shelf First)
                    valid_zones.sort(key=lambda z: z.get('z1', 0))
                    
                    # Zone Selection Heuristic:
                    # Strictly Sequential: 0, 1, 2, ... looping if needed
                    zone_idx = attempt % len(valid_zones)
                    
                    # Safety clamp
                    zone_idx = min(zone_idx, len(valid_zones) - 1)
                    zone = valid_zones[zone_idx]
                    
                    zone_z1 = zone.get('z1', 0)
                    zone_z2 = zone.get('z2', wh_hgt)
                    
                    min_x = zone['x1'] + dim_x / 2
                    max_x = zone['x2'] - dim_x / 2
                    min_y = zone['y1'] + dim_y / 2
                    max_y = zone['y2'] - dim_y / 2
                    
                    if max_x < min_x: max_x = min_x = (zone['x1'] + zone['x2']) / 2
                    if max_y < min_y: max_y = min_y = (zone['y1'] + zone['y2']) / 2
                    
                    # Dense Pact Heuristic (Zone)
                    if attempt < 5:
                        x = min_x
                        y = min_y
                    elif attempt < 45 and len(other_bbox) > 0:
                        rand_idx = random.randint(0, len(other_bbox)-1)
                        ref_box = other_bbox[rand_idx]
                        if random.random() < 0.5:
                            x = ref_box[2] + dim_x / 2
                            y = ref_box[1] + dim_y / 2
                        else:
                            x = ref_box[0] + dim_x / 2
                            y = ref_box[3] + dim_y / 2
                        x += random.uniform(-1, 1)
                        y += random.uniform(-1, 1)
                        x = max(min_x, min(max_x, x))
                        y = max(min_y, min(max_y, y))
                    else:
                        x = round(random.uniform(min_x, max_x))
                        y = round(random.uniform(min_y, max_y))
                else:
                    min_x = dim_x / 2
                    max_x = wh_len - dim_x / 2
                    min_y = dim_y / 2
                    max_y = wh_wid - dim_y / 2
                    
                    if max_x < min_x: max_x = min_x
                    if max_y < min_y: max_y = min_y
                    
                    # Dense Pact Heuristic (Global)
                    if attempt < 3:
                         x = min_x
                         y = min_y
                    elif attempt < 6 and len(other_bbox) > 0:
                         rand_idx = random.randint(0, len(other_bbox)-1)
                         ref_box = other_bbox[rand_idx]
                         if random.random() < 0.5:
                             x = ref_box[2] + dim_x / 2
                             y = ref_box[1] + dim_y / 2
                         else:
                             x = ref_box[0] + dim_x / 2
                             y = ref_box[3] + dim_y / 2

                         x = max(min_x, min(max_x, x))
                         y = max(min_y, min(max_y, y))
                    else:
                         x = round(random.uniform(min_x, max_x))
                         y = round(random.uniform(min_y, max_y))
            else:
                min_x = dim_x / 2
                max_x = wh_len - dim_x / 2
                min_y = dim_y / 2
                max_y = wh_wid - dim_y / 2
                
                if max_x < min_x: max_x = min_x
                if max_y < min_y: max_y = min_y
                
                x = round(random.uniform(min_x, max_x))
                y = round(random.uniform(min_y, max_y))
            
            z = calculate_z_for_item(x, y, dim_x + 0.001, dim_y + 0.001, other_bbox, other_z, other_h)
            
            # Enforce Layer Floor
            z = max(z, zone_z1)
            
            # Layer Snap: If item protrudes through ceiling of current layer, snap to next layer floor
            if z + item_hgt > zone_z2 and zone_z2 < wh_hgt:
                z = zone_z2
            
            if z < best_z:
                best_x, best_y, best_z, best_rotation = x, y, z, rotation
            
            if z == zone_z1:
                break
        
        if best_z > 50000:
            best_z -= 100000.0
        
        x, y, z = best_x, best_y, best_z
        rotation = best_rotation
        solution[item_to_mutate] = [x, y, z, rotation]
        
    return solution

def process_offspring_batch(parent1, parent2, crossover_rate, mutation_rate, wh_dims=None, items_props=None, valid_z=None, allocation_zones=None):
    c1, c2 = crossover_numpy(parent1, parent2, crossover_rate)
    
    # helper to repair with explicit args or None (which will pick up globals)
    m1 = mutation_numpy(c1, wh_dims, items_props, valid_z, mutation_rate, allocation_zones)
    m2 = mutation_numpy(c2, wh_dims, items_props, valid_z, mutation_rate, allocation_zones)
    
    m1 = repair_solution_compact(m1, items_props, wh_dims, allocation_zones, valid_z)
    m2 = repair_solution_compact(m2, items_props, wh_dims, allocation_zones, valid_z)
    
    return [m1, m2]


def eo_evaluate_mutation(solution, item_to_mutate, items_props, wh_dims, valid_z, allocation_zones, 
                         exclusion_zones_arr, current_weights, seed_x, seed_y, seed_z, rotation):
    """
    Evaluate a single EO mutation candidate. Used for parallel evaluation.
    
    Args:
        solution: Base solution array (N, 4)
        item_to_mutate: Index of item to mutate
        items_props: Item properties array (N, 8)
        wh_dims: Warehouse dimensions tuple
        valid_z: Valid z positions array
        allocation_zones: List of allocation zone dicts or None
        exclusion_zones_arr: Exclusion zones array or None
        current_weights: Fitness weights dict
        seed_x, seed_y, seed_z, rotation: The mutation seed position and rotation
        
    Returns:
        (mutated_solution, fitness, space_util, accessibility, stability)
    """
    # Create copy to avoid modifying original
    mutated_sol = solution.copy()
    
    # Apply mutation
    mutated_sol[item_to_mutate] = [seed_x, seed_y, seed_z, rotation]
    
    # Repair solution
    mutated_sol = repair_solution_compact(mutated_sol, items_props, wh_dims, allocation_zones, valid_z)
    
    # Evaluate fitness
    fitness, su, acc, sta, grp = fitness_function_numpy(
        solution=mutated_sol, items_props=items_props, warehouse_dims=wh_dims,
        weights=current_weights, valid_z=valid_z, exclusion_zones_arr=exclusion_zones_arr
    )
    
    return (mutated_sol, fitness, su, acc, sta)


class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=500, crossover_rate=0.8,
                 mutation_rate=0.1, selection_method='tournament'):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.initial_population_from_solution = None

    def initialize_population(self, num_items, warehouse_dims, items_props, valid_z=None, allocation_zones=None):
        # Population shape: (pop_size, num_items, 4) -> x, y, z, rotation
        print(f"Initializing population of size {self.population_size}...", flush=True)
        
        population = []
        
        # Use singleton pool
        pool = get_process_pool()
        process_count = pool._processes if hasattr(pool, '_processes') else multiprocessing.cpu_count()
        
        if process_count > 1:
             # Explicitly pass all data args
             args = [(num_items, warehouse_dims, items_props, allocation_zones, valid_z) for _ in range(self.population_size)]
             
             print(f"Starting parallel initialization with {process_count} processes...")
             population = pool.starmap(create_and_repair, args)
        else:
             print("Starting serial initialization...")
             # Serial execution
             for _ in range(self.population_size):
                 population.append(create_and_repair(num_items, warehouse_dims, items_props, allocation_zones, valid_z))

        return np.array(population)

    def _get_rotated_bounding_box_dims(self, length, width, rotation):
        # Simple lookup since rotation is 0, 90, 180, 270
        # If rotation is integer:
        # 0, 180 -> same
        # 90, 270 -> swapped
        if int(rotation) % 180 == 0:
            return width, length # Function returns width, height logic in original code
        else:
            return length, width

    # create_random_solution_array moved to standalone function


    # Optimized collision check NOT used in full batch init for speed,
    # but used during mutation/repair
    
    # fitness_function_numpy moved to standalone function
    



    def selection(self, population, fitness_scores):
        # Numpy-based tournament selection
        pop_size = len(population)
        indices = np.arange(pop_size)
        selected_indices = []
        
        for _ in range(pop_size):
            # Sample 3
            tourn_idx = np.random.choice(indices, 3, replace=False)
            best_idx = tourn_idx[np.argmax([fitness_scores[i] for i in tourn_idx])]
            selected_indices.append(best_idx)
            
        return population[selected_indices]

    # Crossover and Mutation moved to global functions for multiprocessing


    def optimize(self, items, warehouse, weights=None, callback=None, optimization_state=None, initial_solution=None):
        num_items = len(items)
        if num_items == 0:
            return [], 0, 0
        
        import time
        start_time = time.time()
            
        # Get zones
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
             ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
             if ex_zones:
                 exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
            
        # Pre-process items into numpy array (N, 9)
        items_props = np.zeros((num_items, 9), dtype=np.float32)
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
            
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        
        unstackable_count = np.sum(items_props[:, 4] == 0)
        with open('thread_debug.log', 'a') as f:
             f.write(f"DEBUG: Unstackable Items Count: {unstackable_count} / {num_items}\n")

        valid_z = get_valid_z_positions(warehouse)
        
        # Get allocation zones for constraining item placement
        allocation_zones = None
        if zones:
            alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
            if alloc_zones:
                allocation_zones = alloc_zones
        
        # Parallel initialization
        if callback:
            callback(0, 0, 0, None, 0, 0, 0, message="Initializing population... (This may take a moment)")
        
        population = self.initialize_population(num_items, wh_dims, items_props, valid_z, allocation_zones)
        
        if initial_solution is not None:
             # Convert to numpy (N, 4)
             sol_arr = np.zeros((num_items, 4), dtype=np.float32)
             item_map = {item['id']: i for i, item in enumerate(items)}
             for sol_item in initial_solution:
                 idx = item_map.get(sol_item['id'])
                 if idx is not None:
                     sol_arr[idx] = [sol_item['x'], sol_item['y'], sol_item['z'], sol_item['rotation']]
             
             # Seed into population
             population[0] = sol_arr
        
        best_solution = None
        best_fitness = -float('inf')
        start_time = time.time()
        time_to_best = 0
        
        # Use singleton pool to prevent process churn
        pool = get_process_pool()
        process_count = pool._processes if hasattr(pool, '_processes') else multiprocessing.cpu_count()
        
        try:
            for generation in range(self.generations):
                if optimization_state and not optimization_state['running']:
                    break
                    
                fitness_scores = []
                metrics_list = []
                
                if callback:
                     msg = f"GA Generation {generation + 1}/{self.generations}"
                     callback((generation) / self.generations * 100, best_fitness if best_fitness != -float('inf') else 0, best_fitness if best_fitness != -float('inf') else 0, None, 0, 0, 0, message=msg)

                current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
                
                # Parallel Fitness Evaluation
                if process_count > 1:
                     # Explicitly pass all data args to avoid globals
                     fit_args = [(sol, items_props, wh_dims, current_weights, valid_z, exclusion_zones_arr) for sol in population]
                     results = pool.starmap(fitness_function_numpy, fit_args)
                     
                     results_arr = np.array(results)
                     fitness_scores = results_arr[:, 0]
                     metrics_list = results_arr[:, 1:]
                else:
                    # Serial Fallback for debugging (or if pool fails)
                    for sol in population:
                        f, su, acc, sta, grp = fitness_function_numpy(
                            sol, items_props, wh_dims, current_weights, valid_z, exclusion_zones_arr)
                        fitness_scores.append(f)
                        metrics_list.append((su, acc, sta, grp))
                    metrics_list = np.array(metrics_list)
                    
                fitness_scores = np.array(fitness_scores)
                best_idx = np.argmax(fitness_scores)
                
                if fitness_scores[best_idx] > best_fitness:
                    best_fitness = fitness_scores[best_idx]
                    best_solution = population[best_idx].copy()
                    time_to_best = time.time() - start_time
                    
                # Selection
                selected_pop = self.selection(population, fitness_scores)
                
                # Crossover & Mutation & Repair (Parallelized)
                new_pop = []
                
                if process_count > 1:
                     offspring_args = []
                     for i in range(0, len(selected_pop), 2):
                         if i+1 < len(selected_pop):
                              p1 = selected_pop[i]
                              p2 = selected_pop[i+1]
                              offspring_args.append((p1, p2, self.crossover_rate, self.mutation_rate, wh_dims, items_props, valid_z, allocation_zones))
                         else:
                              # Self-crossover? (Copy really)
                              offspring_args.append((selected_pop[i], selected_pop[i], self.crossover_rate, self.mutation_rate, wh_dims, items_props, valid_z, allocation_zones))
                     
                     if offspring_args:
                         results = pool.starmap(process_offspring_batch, offspring_args)
                         for pair in results:
                             new_pop.extend(pair)
                else:
                    for i in range(0, len(selected_pop), 2):
                        if i+1 < len(selected_pop):
                             batch_res = process_offspring_batch(selected_pop[i], selected_pop[i+1], self.crossover_rate, self.mutation_rate, wh_dims, items_props, valid_z, allocation_zones)
                             new_pop.extend(batch_res)
                        else:
                             batch_res = process_offspring_batch(selected_pop[i], selected_pop[i], self.crossover_rate, self.mutation_rate, wh_dims, items_props, valid_z, allocation_zones)
                             new_pop.extend(batch_res)
                
                new_pop = new_pop[:self.population_size]
                population = np.array(new_pop)
                
                # Callback update
                if callback:
                    progress = (generation + 1) / self.generations * 100
                    avg_fitness = np.mean(fitness_scores)
                    avg_space = np.mean(metrics_list[:, 0])
                    avg_acc = np.mean(metrics_list[:, 1])
                    avg_stab = np.mean(metrics_list[:, 2])
                    
                    converted_solution = None
                    if best_solution is not None:
                        try:
                            converted_solution = []
                            for i in range(num_items):
                                converted_solution.append({
                                    'id': items[i]['id'],
                                    'x': float(best_solution[i, 0]),
                                    'y': float(best_solution[i, 1]),
                                    'z': float(best_solution[i, 2]),
                                    'rotation': int(best_solution[i, 3])
                                })
                        except:
                            converted_solution = None
                    
                    callback(progress, avg_fitness, best_fitness, converted_solution,
                             avg_space, avg_acc, avg_stab)

                if generation % 10 == 0:
                     gc.collect()
        except Exception as e:
            print(f"Optimization Loop Error: {e}")
            raise e

        # Final conversion
        final_sol_list = []
        if best_solution is not None:
            for i in range(num_items):
                final_sol_list.append({
                    'id': items[i]['id'],
                    'x': float(best_solution[i, 0]),
                    'y': float(best_solution[i, 1]),
                    'z': float(best_solution[i, 2]),
                    'rotation': int(best_solution[i, 3])
                })
                
        return final_sol_list, best_fitness, time_to_best

    # Method to satisfy app.py's direct call to fitness_function for metrics
    def fitness_function(self, solution_list, items, warehouse, weights=None):
        # Convert list of dicts back to numpy for calc
        num_items = len(items)
        # Create map
        item_map = {item['id']: i for i, item in enumerate(items)}
        
        sol_array = np.zeros((num_items, 4))
        # Default all zeros
        
        for item_sol in solution_list:
            idx = item_map.get(item_sol['id'])
            if idx is not None:
                sol_array[idx] = [item_sol['x'], item_sol['y'], item_sol['z'], item_sol['rotation']]
                
        items_props = np.zeros((num_items, 9))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
            
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        valid_z = get_valid_z_positions(warehouse)
        
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
             ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
             if ex_zones:
                 exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])

        return fitness_function_numpy(sol_array, items_props, wh_dims, 
            weights or {'space': 0.5}, valid_z, exclusion_zones_arr)


        
    def calculate_center_of_gravity(self, solution_list, items_dict):
        # Original logic adaptable
        total_mass = 0
        mx, my, mz = 0, 0, 0
        for sol in solution_list:
            item = items_dict.get(sol['id'])
            if item:
                mass = item.get('weight', 0)
                mx += mass * sol['x']
                my += mass * sol['y']
                mz += mass * (sol['z'] + item['height']/2)
                total_mass += mass
        if total_mass == 0: return 0,0,0
        return mx/total_mass, my/total_mass, mz/total_mass


class ExtremalOptimization:
    """Extremal Optimization algorithm - focuses on improving worst-performing items."""
    def __init__(self, iterations=1000, tau=1.5):
        self.iterations = iterations
        self.tau = tau

    def optimize(self, items, warehouse, weights=None, callback=None, optimization_state=None):
        num_items = len(items)
        if num_items == 0:
            return [], 0, 0
            
        import time
        start_time = time.time()
        
        # Get zones
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
            ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
            if ex_zones:
                exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
            
        # Pre-process items into numpy array
        items_props = np.zeros((num_items, 9))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
            
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        valid_z = get_valid_z_positions(warehouse)
        
        # Get allocation zones for constraining item placement
        allocation_zones = None
        if zones:
            alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
            if alloc_zones:
                allocation_zones = alloc_zones
        
        # Initialize single solution using GA's helper
        solution = create_random_solution_array(num_items, wh_dims, items_props, allocation_zones)
        solution = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, valid_z)
        
        best_solution = solution.copy()
        current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
        best_fitness, _, _, _, _ = fitness_function_numpy(
            solution=solution, items_props=items_props, warehouse_dims=wh_dims, 
            weights=current_weights, valid_z=valid_z, exclusion_zones_arr=exclusion_zones_arr)

        
        start_time = time.time()
        time_to_best = 0
        
        # Use singleton pool for parallel evaluation
        pool = get_process_pool()
        process_count = pool._processes if hasattr(pool, '_processes') else multiprocessing.cpu_count()
        num_parallel_candidates = max(4, process_count)  # Evaluate multiple candidates per iteration
        
        for iteration in range(self.iterations):
            if optimization_state and not optimization_state['running']:
                break
                
            # Calculate individual item fitnesses (local contribution) - vectorized
            door_x = wh_dims[3] if len(wh_dims) >= 5 else 0
            door_y = wh_dims[4] if len(wh_dims) >= 5 else 0
            
            # Vectorized item fitness calculation
            dists = np.sqrt((solution[:, 0] - door_x)**2 + (solution[:, 1] - door_y)**2)
            item_fitnesses = (1.0 / (1.0 + dists)) * items_props[:, 5]
            
            # Find worst items using power-law selection (tau parameter)
            # Rank items by fitness (lower = worse)
            ranks = np.argsort(item_fitnesses)  # Indices sorted by fitness (ascending = worst first)
            
            # Select item to mutate using power-law: P(rank) ~ rank^(-tau)
            n = len(ranks)
            probs = np.arange(1, n + 1, dtype=float) ** (-self.tau)
            probs /= probs.sum()
            
            # Select one of the worst items to mutate
            selected_rank_idx = np.random.choice(n, p=probs)
            item_to_mutate = ranks[selected_rank_idx]
            
            # Mutate the selected item (try new random positions in parallel)
            old_pos = solution[item_to_mutate].copy()
            
            item_len = items_props[item_to_mutate, 0]
            item_wid = items_props[item_to_mutate, 1]
            can_rotate = items_props[item_to_mutate, 3]
            wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
            has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
            
            # Generate multiple candidate mutations for parallel evaluation
            candidate_args = []
            for _ in range(num_parallel_candidates):
                # Determine rotation
                rotation = old_pos[3]
                if can_rotate and random.random() > 0.7:
                    rotation = random.choice([0, 90, 180, 270])
                
                if int(rotation) % 180 == 0:
                    dx, dy = item_len, item_wid
                else:
                    dx, dy = item_wid, item_len
                
                # Generate seed position
                if has_allocation_zones:
                    seed_zone = random.choice(allocation_zones)
                    z_min_x, z_max_x = seed_zone['x1'], seed_zone['x2']
                    z_min_y, z_max_y = seed_zone['y1'], seed_zone['y2']
                    
                    safe_min_x = z_min_x + dx/2
                    safe_max_x = z_max_x - dx/2
                    safe_min_y = z_min_y + dy/2
                    safe_max_y = z_max_y - dy/2
                    
                    if safe_max_x < safe_min_x: safe_max_x = safe_min_x = (z_min_x + z_max_x)/2
                    if safe_max_y < safe_min_y: safe_max_y = safe_min_y = (z_min_y + z_max_y)/2
                    
                    seed_x = random.uniform(safe_min_x, safe_max_x)
                    seed_y = random.uniform(safe_min_y, safe_max_y)
                    seed_z = seed_zone.get('z1', 0)
                else:
                    seed_x = random.uniform(dx/2, wh_len - dx/2)
                    seed_y = random.uniform(dy/2, wh_wid - dy/2)
                    seed_z = 0
                
                candidate_args.append((
                    solution, item_to_mutate, items_props, wh_dims, valid_z, 
                    allocation_zones, exclusion_zones_arr, current_weights,
                    seed_x, seed_y, seed_z, rotation
                ))
            
            # Evaluate candidates in parallel
            if process_count > 1 and len(candidate_args) > 1:
                results = pool.starmap(eo_evaluate_mutation, candidate_args)
            else:
                # Serial fallback
                results = [eo_evaluate_mutation(*args) for args in candidate_args]
            
            # Find best candidate
            best_candidate_idx = -1
            best_candidate_fitness = best_fitness
            for i, (mutated_sol, fitness, su, acc, sta) in enumerate(results):
                if fitness > best_candidate_fitness:
                    best_candidate_fitness = fitness
                    best_candidate_idx = i
            
            # Accept best if it improved
            if best_candidate_idx >= 0:
                best_sol_result = results[best_candidate_idx]
                solution = best_sol_result[0]
                new_fitness = best_sol_result[1]
                su, acc, sta = best_sol_result[2], best_sol_result[3], best_sol_result[4]
                best_fitness = new_fitness
                best_solution = solution.copy()
                time_to_best = time.time() - start_time
            else:
                # Keep old position - no improvement found
                new_fitness = best_fitness
                su, acc, sta = 0, 0, 0
        
            # Callback for progress updates
            if callback and (iteration % 10 == 0 or iteration == self.iterations - 1):
                progress = (iteration + 1) / self.iterations * 100
                
                converted_solution = None
                if True: # Always convert if we are calling callback
                    converted_solution = []
                    for i in range(num_items):
                        converted_solution.append({
                            'id': items[i]['id'],
                            'x': float(best_solution[i, 0]),
                            'y': float(best_solution[i, 1]),
                            'z': float(best_solution[i, 2]),
                            'rotation': int(best_solution[i, 3])
                        })
                
                msg = f"EO Iteration {iteration + 1}/{self.iterations} - Best Fit: {best_fitness:.4f}"
                callback(progress, new_fitness, best_fitness, converted_solution, su, acc, sta, message=msg)
        
        # Final conversion
        final_sol_list = []
        for i in range(num_items):
            final_sol_list.append({
                'id': items[i]['id'],
                'x': float(best_solution[i, 0]),
                'y': float(best_solution[i, 1]),
                'z': float(best_solution[i, 2]),
                'rotation': int(best_solution[i, 3])
            })
        
        return final_sol_list, best_fitness, time_to_best


class HybridOptimizer:
    """Hybrid optimizer combining GA global search with EO local refinement."""
    def __init__(self, ga_generations=500, eo_iterations=1000, population_size=100):
        self.ga_generations = ga_generations
        self.eo_iterations = eo_iterations
        self.population_size = population_size
        
    def optimize(self, items, warehouse, weights=None, callback=None, optimization_state=None):
        num_items = len(items)
        if num_items == 0:
            return [], 0, 0
        
        import time
        start_time = time.time()
        
        # Phase 1: Run GA for global exploration (70% of progress)
        def ga_callback(progress, avg_fit, best_fit, solution, space, access, stability, message=None):
            if callback:
                # Scale GA progress to 0-70%
                msg = message if message else "Hybrid Phase 1: Genetic Algorithm"
                callback(progress * 0.7, avg_fit, best_fit, solution, space, access, stability, message=msg)


        
        ga = GeneticAlgorithm(generations=self.ga_generations, population_size=self.population_size)
        ga_solution, ga_fitness, ga_time_to_best = ga.optimize(
            items, warehouse, weights, ga_callback, optimization_state
        )
        
        if optimization_state and not optimization_state['running']:
            return ga_solution, ga_fitness, ga_time_to_best
        
        # Phase 2: Run EO for local refinement using GA's best solution as seed
        # Convert GA solution back to numpy for EO to use
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
            ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
            if ex_zones:
                exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
        
        items_props = np.zeros((num_items, 9))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000,
                item.get('fragility', 0)
            ]
        
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        valid_z = get_valid_z_positions(warehouse)
        
        # Get allocation zones for constraining item placement
        allocation_zones = None
        if zones:
            alloc_zones = [z for z in zones if z['zone_type'] == 'allocation']
            if alloc_zones:
                allocation_zones = alloc_zones
        
        # Convert GA solution list to numpy array
        item_id_to_idx = {item['id']: i for i, item in enumerate(items)}
        solution = np.zeros((num_items, 4))
        for sol_item in ga_solution:
            idx = item_id_to_idx.get(sol_item['id'])
            if idx is not None:
                solution[idx] = [sol_item['x'], sol_item['y'], sol_item['z'], sol_item['rotation']]
        
        best_solution = solution.copy()
        best_fitness = ga_fitness
        time_to_best = ga_time_to_best
        current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
        
        ga_helper = GeneticAlgorithm()
        
        # Use singleton pool for parallel EO evaluation
        pool = get_process_pool()
        process_count = pool._processes if hasattr(pool, '_processes') else multiprocessing.cpu_count()
        num_parallel_candidates = max(4, process_count)
        
        # EO refinement phase (with parallel candidate evaluation)
        for iteration in range(self.eo_iterations):
            if optimization_state and not optimization_state['running']:
                break
            
            # Calculate individual item fitnesses - vectorized
            door_x = wh_dims[3] if len(wh_dims) >= 5 else 0
            door_y = wh_dims[4] if len(wh_dims) >= 5 else 0
            dists = np.sqrt((solution[:, 0] - door_x)**2 + (solution[:, 1] - door_y)**2)
            item_fitnesses = (1.0 / (1.0 + dists)) * items_props[:, 5]
            
            # Power-law selection for worst items
            ranks = np.argsort(item_fitnesses)
            n = len(ranks)
            tau = 1.5
            probs = np.arange(1, n + 1, dtype=float) ** (-tau)
            probs /= probs.sum()
            
            selected_rank_idx = np.random.choice(n, p=probs)
            item_to_mutate = ranks[selected_rank_idx]
            
            old_pos = solution[item_to_mutate].copy()
            item_len = items_props[item_to_mutate, 0]
            item_wid = items_props[item_to_mutate, 1]
            can_rotate = items_props[item_to_mutate, 3]
            wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
            has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
            
            # Generate multiple candidate mutations for parallel evaluation
            candidate_args = []
            for _ in range(num_parallel_candidates):
                rotation = old_pos[3]
                if can_rotate and random.random() > 0.7:
                    rotation = random.choice([0, 90, 180, 270])
                
                if int(rotation) % 180 == 0:
                    dx, dy = item_len, item_wid
                else:
                    dx, dy = item_wid, item_len
                
                if has_allocation_zones:
                    seed_zone = random.choice(allocation_zones)
                    z_min_x, z_max_x = seed_zone['x1'], seed_zone['x2']
                    z_min_y, z_max_y = seed_zone['y1'], seed_zone['y2']
                    
                    safe_min_x = z_min_x + dx/2
                    safe_max_x = z_max_x - dx/2
                    safe_min_y = z_min_y + dy/2
                    safe_max_y = z_max_y - dy/2
                    
                    if safe_max_x < safe_min_x: safe_max_x = safe_min_x = (z_min_x + z_max_x)/2
                    if safe_max_y < safe_min_y: safe_max_y = safe_min_y = (z_min_y + z_max_y)/2
                    
                    seed_x = random.uniform(safe_min_x, safe_max_x)
                    seed_y = random.uniform(safe_min_y, safe_max_y)
                    seed_z = seed_zone.get('z1', 0)
                else:
                    seed_x = random.uniform(dx/2, wh_len - dx/2)
                    seed_y = random.uniform(dy/2, wh_wid - dy/2)
                    seed_z = 0
                
                candidate_args.append((
                    solution, item_to_mutate, items_props, wh_dims, valid_z,
                    allocation_zones, exclusion_zones_arr, current_weights,
                    seed_x, seed_y, seed_z, rotation
                ))
            
            # Evaluate candidates in parallel
            if process_count > 1 and len(candidate_args) > 1:
                results = pool.starmap(eo_evaluate_mutation, candidate_args)
            else:
                results = [eo_evaluate_mutation(*args) for args in candidate_args]
            
            # Find best candidate
            best_candidate_idx = -1
            best_candidate_fitness = best_fitness
            for i, (mutated_sol, fitness, su, acc, sta) in enumerate(results):
                if fitness > best_candidate_fitness:
                    best_candidate_fitness = fitness
                    best_candidate_idx = i
            
            # Accept best if it improved
            if best_candidate_idx >= 0:
                best_sol_result = results[best_candidate_idx]
                solution = best_sol_result[0]
                new_fitness = best_sol_result[1]
                su, acc, sta = best_sol_result[2], best_sol_result[3], best_sol_result[4]
                best_fitness = new_fitness
                best_solution = solution.copy()
                time_to_best = time.time() - start_time
            else:
                new_fitness = best_fitness
                su, acc, sta = 0, 0, 0
            
            # Callback for progress updates (70-100%)
            if callback and (iteration % 10 == 0 or iteration == self.eo_iterations - 1):
                eo_progress = (iteration + 1) / self.eo_iterations
                total_progress = 70 + eo_progress * 30  # 70-100%
                
                converted_solution = None
                if True: # Always convert if we are calling callback
                    converted_solution = []
                    for i in range(num_items):
                        converted_solution.append({
                            'id': items[i]['id'],
                            'x': float(best_solution[i, 0]),
                            'y': float(best_solution[i, 1]),
                            'z': float(best_solution[i, 2]),
                            'rotation': int(best_solution[i, 3])
                        })
                
                msg = f"Hybrid Phase 2: EO Iteration {iteration + 1}/{self.eo_iterations}"
                callback(total_progress, new_fitness, best_fitness, converted_solution, su, acc, sta, message=msg)

        
        # Final conversion
        final_sol_list = []
        for i in range(num_items):
            final_sol_list.append({
                'id': items[i]['id'],
                'x': float(best_solution[i, 0]),
                'y': float(best_solution[i, 1]),
                'z': float(best_solution[i, 2]),
                'rotation': int(best_solution[i, 3])
            })
        
        return final_sol_list, best_fitness, time_to_best

    def optimize_eo_ga(self, items, warehouse, weights=None, callback=None, optimization_state=None):
        num_items = len(items)
        if num_items == 0:
            return [], 0, 0
        
        import time
        start_time = time.time()
        
        # Phase 1: Run EO for initial solution (70% of progress)
        def eo_callback(progress, avg_fit, best_fit, solution, space, access, stability, message=None):
            if callback:
                # Scale EO progress to 0-70%
                msg = message if message else "Hybrid Phase 1: Extremal Optimization"
                callback(progress * 0.7, avg_fit, best_fit, solution, space, access, stability, message=msg)

                
        eo = ExtremalOptimization(iterations=self.eo_iterations)
        eo_solution_list, eo_fitness, eo_time = eo.optimize(
            items, warehouse, weights, eo_callback, optimization_state
        )
        
        if optimization_state and not optimization_state['running']:
            return eo_solution_list, eo_fitness, eo_time
            
        # Phase 2: Run GA for refinement using EO solution as seed (30% of progress)
        def ga_callback(progress, avg_fit, best_fit, solution, space, access, stability, message=None):
            if callback:
                # Scale GA progress to 70-100%
                total = 70 + (progress * 0.3)
                msg = message if message else "Hybrid Phase 2: Genetic Algorithm"
                callback(total, avg_fit, best_fit, solution, space, access, stability, message=msg)

        
        ga = GeneticAlgorithm(generations=self.ga_generations, population_size=self.population_size)
        final_solution, final_fitness, final_time = ga.optimize(
             items, warehouse, weights, ga_callback, optimization_state, initial_solution=eo_solution_list
        )
        
        total_time = time.time() - start_time
        return final_solution, final_fitness, total_time
