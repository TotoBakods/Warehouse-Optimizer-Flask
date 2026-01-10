import math
import random
import numpy as np
import time
import multiprocessing
import gc
from functools import partial
from database import get_exclusion_zones

# --- Global Shared Memory for Multiprocessing ---
_pool_items_props = None
_pool_wh_dims = None
_pool_valid_z = None
_pool_allocation_zones = None
_pool_exclusion_zones = None

def init_worker(items_props, wh_dims, valid_z, allocation_zones, exclusion_zones):
    """Initialize worker process with shared read-only data."""
    global _pool_items_props, _pool_wh_dims, _pool_valid_z, _pool_allocation_zones, _pool_exclusion_zones
    _pool_items_props = items_props
    _pool_wh_dims = wh_dims
    _pool_valid_z = valid_z
    _pool_allocation_zones = allocation_zones
    _pool_exclusion_zones = exclusion_zones


# Helper for gravity calculation
def calculate_z_for_item(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h, other_items_stackable=None, strict_stacking=True):
    """
    Calculate the lowest valid Z position for an item given its position and other items.
    
    Args:
        x, y: Center coordinates of the new item
        dim_x, dim_y: Dimensions of the new item
        other_items_bbox: (N, 4) array of [min_x, min_y, max_x, max_y] for placed items
        other_items_z: (N,) array of Z positions for placed items
        other_items_h: (N,) array of heights for placed items
        other_items_stackable: (N,) boolean/int array indicating if item can support others.
    """
    if len(other_items_bbox) == 0:
        return 0.0
        
    # New item bounding box
    new_min_x = x - dim_x / 2
    new_max_x = x + dim_x / 2
    new_min_y = y - dim_y / 2
    new_max_y = y + dim_y / 2
    
    # Check overlaps in XY plane (vectorized)
    # Overlap condition: not (A.left > B.right or A.right < B.left or A.top < B.bottom or A.bottom > B.top)
    # Inverted: (A.left < B.right and A.right > B.left and A.top > B.bottom and A.bottom < B.top)
    # Here Y is depth, so "top/bottom" refers to Y-axis
    
    overlaps_x = (new_min_x < other_items_bbox[:, 2]) & (new_max_x > other_items_bbox[:, 0])
    overlaps_y = (new_min_y < other_items_bbox[:, 3]) & (new_max_y > other_items_bbox[:, 1])
    overlaps = overlaps_x & overlaps_y
    
    if not np.any(overlaps):
        return 0.0
        
    # Get max Z top of overlapping items
    overlapping_z_tops = other_items_z[overlaps] + other_items_h[overlaps]

    # Check if any overlapping item is NOT stackable
    if strict_stacking and other_items_stackable is not None:
        overlapping_stackables = other_items_stackable[overlaps]
        if np.any(overlapping_stackables == 0):
             return 1000000.0 # Effectively impossible
             
    max_z = np.max(overlapping_z_tops)
    
    # Stability Check: Ensure support area is sufficient
    if max_z > 0:
        # Identify supporting items (within small tolerance of max_z)
        is_support = np.abs(overlapping_z_tops - max_z) < 0.01
        
        # Get indices of supports relative to the 'overlaps' subset? NO.
        # 'overlaps' is a boolean mask. 
        # overlapping_z_tops is filtered by [overlaps].
        # is_support is boolean mask matching overlapping_z_tops.
        # We need original indices to get bboxes.
        
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
        if supported_area < (item_area * 0.2): # 20% Support Threshold
            return max_z + 100000.0
            
    return max_z


def repair_solution_compact(solution, items_props=None, warehouse_dims=None, allocation_zones=None, layer_heights=None):
    # Use globals if in worker process
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    if layer_heights is None:
         # layer_heights logic is usually derived from warehouse_dims if not passed contextually.
         # But usually it's passed or can be derived.
         # For safety, if we have warehouse_dims, we can try to derive it via get_valid_z_positions logic?
         # Or just assume it's passed or default to [0.0].
         pass

    """
    Re-calculates positions to enforce gravity and compactness (Tetris-like).
    Packs items towards the origin (0,0,0) by sliding them Left and Back.
    Respects allocation zones and layer heights.
    
    Args:
        solution: (N, 4) array [x, y, z, rotation]
        items_props: (N, 8) or (N, 7) array [length, width, height, can_rotate, ...]
        warehouse_dims: tuple (length, width, height, door_x, door_y) or None
        allocation_zones: list of zone dicts with x1, y1, x2, y2, z1, z2 or None
        layer_heights: list of valid Z positions for layers or None
    """
    num_items = len(solution)
    if num_items == 0:
        return solution
    
    # Default warehouse bounds if not provided
    wh_len = warehouse_dims[0] if warehouse_dims else 100
    wh_wid = warehouse_dims[1] if warehouse_dims else 100
    wh_hgt = warehouse_dims[2] if warehouse_dims else 10
    
    # Default layer heights (just floor if not specified)
    if layer_heights is None or len(layer_heights) == 0:
        layer_heights = [0.0]
    layer_heights = sorted(layer_heights)
    
    # Sort indices:
    # Primary Key: Volume (Descending) - Critical for packing density (Large items first)
    # Secondary Key: Input Z coordinate (Ascending) - Allows GA to shuffle similar items to avoid stagnation
    volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]
    # np.lexsort sorts by keys in reverse order (last key is primary)
    # Keys: (Secondary: Z, Primary: -Volume)
    sorted_indices = np.lexsort((solution[:, 2], -volumes))
    
    # Arrays to track placed items
    placed_bbox = np.zeros((num_items, 4))  # x1, y1, x2, y2
    placed_z = np.zeros(num_items)
    placed_h = np.zeros(num_items)
    placed_stackable = np.zeros(num_items, dtype=int)
    
    count = 0
    
    for idx in sorted_indices:
        l, w, h = items_props[idx, 0:3]
        can_rotate = items_props[idx, 3]
        stackable = items_props[idx, 4]
        
        # Try all rotation options if rotation is allowed (0, 90 degrees)
        orientations_to_try = [0.0, 90.0] if can_rotate else [solution[idx, 3]]
        
        best_state = None
        best_score = float('inf')
        
        # Search for valid zones
        fitting_zones = []
        if allocation_zones and len(allocation_zones) > 0:
            for zone in allocation_zones:
                zw = zone['x2'] - zone['x1']
                zh = zone['y2'] - zone['y1']
                if min(l, w) <= max(zw, zh) and max(l, w) <= max(zw, zh):
                     fitting_zones.append(zone)
        
        # If no allocation zones provided or none fit, treat whole warehouse as one zone
        if not fitting_zones:
             fitting_zones = [{
                'x1': 0.0, 'y1': 0.0, 'x2': wh_len, 'y2': wh_wid, 
                'z1': 0.0, 'z2': wh_hgt
             }]

        # Try each fitting zone
        # Shuffle zones to ensure balanced distribution (User Request: "not distributed properly")
        random.shuffle(fitting_zones)
        
        for zone in fitting_zones:
            zone_x1 = zone['x1']
            zone_y1 = zone['y1']
            zone_x2 = zone['x2']
            zone_y2 = zone['y2']
            zone_z1 = zone.get('z1', 0.0)
            zone_z2 = zone.get('z2', wh_hgt)

            for rot in orientations_to_try:
                # Calculate Dimensions based on rotation
                if int(rot) % 180 == 0:
                    dx, dy = l, w
                else:
                    dx, dy = w, l
                
                # Skip if item doesn't fit in zone at all (double check)
                if dx > (zone_x2 - zone_x1) or dy > (zone_y2 - zone_y1):
                    continue
                
                # Try each layer, starting from bottom
                found_position = False
                for layer_z in layer_heights:
                    if layer_z < zone_z1 or layer_z >= zone_z2:
                        continue
                    
                    # Calculate layer ceiling (next layer height or zone top)
                    layer_ceiling = zone_z2
                    
                    # Look for the next defined layer
                    next_layer_exists = False
                    for next_lz in layer_heights:
                        if next_lz > layer_z:
                            layer_ceiling = next_lz
                            next_layer_exists = True
                            break
                    
                    # Infere last layer height if no explicit ceiling from next layer
                    if not next_layer_exists and len(layer_heights) > 1:
                        # Find previous layer Z
                        prev_layer_z = layer_heights[0]
                        for lz in layer_heights:
                             if lz < layer_z:
                                 prev_layer_z = lz
                             else:
                                 break
                        
                        inferred_height = layer_z - prev_layer_z
                        if inferred_height > 0:
                            # Apply constraint: min of zone top OR inferred top
                            layer_ceiling = min(zone_z2, layer_z + inferred_height)

                    # Check if item fits within layer height
                    if layer_z + h > layer_ceiling + 0.01:  # Small tolerance
                        continue  # Item too tall for this layer
                    
                    # --- Tetris-like placement: scan for first open position ---
                    step_x = max(0.5, dx / 2)
                    step_y = max(0.5, dy / 2)
                    
                    # Start from zone corner
                    test_x = zone_x1 + dx / 2
                    found_position = False
                    
                    while test_x + dx / 2 <= zone_x2 + 0.01:
                        test_y = zone_y1 + dy / 2
                        
                        while test_y + dy / 2 <= zone_y2 + 0.01:
                            # Check Z at this position
                            # Add small padding to query dimensions to enforce separation/prevent Z-fighting
                            z = calculate_z_for_item(test_x, test_y, dx + 0.001, dy + 0.001, 
                                                     placed_bbox[:count], placed_z[:count], placed_h[:count], placed_stackable[:count])
                            
                            # Apply layer floor constraint
                            z = max(z, layer_z)
                            
                            # Check if item fits within layer ceiling
                            if z + h <= layer_ceiling + 0.01:
                                # Valid position found!
                                # Calculate score: prefer lower Z, then closer to origin (relative to zone? or global?)
                                # Using global coordinates ensures packing towards 0,0 usually.
                                # But we might want to pack towards zone start?
                                # Let's stick to global coordinate minimization for consistency, 
                                # OR minimize Z first.
                                score = z * 10000.0 + test_x + test_y
                                
                                if score < best_score:
                                    best_score = score
                                    best_state = (test_x, test_y, z, rot, dx, dy)
                                    found_position = True
                                    # Optimization: If we found ANY valid spot, 
                                    # and since we scan layers bottom-up, and zones... 
                                    # Should we break strictly? 
                                    # If we break here, we take the first valid spot in the first valid zone.
                                    # This is good for "filling Zone A then Zone B".
                                    # So yes, break.
                            
                            if found_position: break
                            test_y += step_y
                        if found_position: break
                        test_x += step_x
                    if found_position: break
                if found_position: break
            if found_position: break
        
        # If no valid position found in assigned zones, try GLOBAL SPILLOVER (search entire warehouse)
        # [DISABLED] Global Spillover disabled to strict zone enforcement
        if False and best_state is None and allocation_zones and len(allocation_zones) > 0:
            # Spillover Zone: Full Warehouse
            zone_x1, zone_y1, zone_z1 = 0.0, 0.0, 0.0
            zone_x2, zone_y2, zone_z2 = wh_len, wh_wid, wh_hgt
            
            # Reuse logic for one specific global zone
            for rot in orientations_to_try:
                if int(rot) % 180 == 0:
                    dx, dy = l, w
                else:
                    dx, dy = w, l
                
                if dx > wh_len or dy > wh_wid: continue

                # Try each layer
                for layer_z in layer_heights:
                    # Calculate layer ceiling
                    layer_ceiling = wh_hgt
                    
                    # Look for the next defined layer
                    next_layer_exists = False
                    for next_lz in layer_heights:
                        if next_lz > layer_z:
                            layer_ceiling = next_lz
                            next_layer_exists = True
                            break
                    
                    # If this is the last layer (no next layer defined), 
                    # try to infer a height constraint to prevent infinite stacking.
                    # This prevents the "last layer is the whole rest of the warehouse" behavior
                    # which causes infinite stacking if users expect shelves.
                    if not next_layer_exists and len(layer_heights) > 1:
                        # Estimate layer height from the previous interval
                        # (Assuming uniform layer spacing)
                        # Find the layer before this one
                        prev_layer_z = layer_heights[0]
                        for lz in layer_heights:
                             if lz < layer_z:
                                 prev_layer_z = lz
                             else:
                                 break
                        
                        inferred_height = layer_z - prev_layer_z
                        if inferred_height > 0:
                            # Apply this height to the current layer
                            layer_ceiling = min(wh_hgt, layer_z + inferred_height)

                    if layer_z + h > layer_ceiling + 0.01: continue
                    
                    # Tetris scan
                    step_x = max(0.5, dx / 2)
                    step_y = max(0.5, dy / 2)
                    test_x = zone_x1 + dx / 2
                    found_position = False
                    
                    while test_x + dx / 2 <= zone_x2 + 0.01:
                        test_y = zone_y1 + dy / 2
                        while test_y + dy / 2 <= zone_y2 + 0.01:
                            z = calculate_z_for_item(test_x, test_y, dx, dy, 
                                                     placed_bbox[:count], placed_z[:count], placed_h[:count], placed_stackable[:count])
                            z = max(z, layer_z)
                            
                            if z + h <= layer_ceiling + 0.01:
                                score = z * 10000.0 + test_x + test_y
                                if score < best_score:
                                    best_score = score
                                    best_state = (test_x, test_y, z, rot, dx, dy)
                                    found_position = True
                            if found_position: break
                            test_y += step_y
                        if found_position: break
                        test_x += step_x
                    if found_position: break
                if found_position: break

        # If STILL no valid position found, use fallback (using the FIRST fitting zone as default container)
        if best_state is None:
            # Debug log
            # Debug log removed

            
            # Pick first fitting zone as default or global if none
            zone = fitting_zones[0] if fitting_zones else {'x1':0, 'y1':0, 'x2':wh_len, 'y2':wh_wid}
            zone_x1, zone_y1 = zone.get('x1', 0), zone.get('y1', 0)
            zone_x2, zone_y2 = zone.get('x2', wh_len), zone.get('y2', wh_wid)
            zone_z1, zone_z2 = zone.get('z1', 0.0), zone.get('z2', wh_hgt)

            fallback_rot = solution[idx, 3]
            if int(fallback_rot) % 180 == 0:
                dx, dy = l, w
            else:
                dx, dy = w, l
            
        if best_state is None:
             # Fallback: Try to find ANY valid position
             # Try random positions to avoid unstackable items
             found_fallback = False
             
             # First try original position clamped
             candidates = [(max(zone_x1 + dx/2, min(zone_x2 - dx/2, solution[idx, 0])),
                            max(zone_y1 + dy/2, min(zone_y2 - dy/2, solution[idx, 1])))]
                            
             # Add random candidates
             for _ in range(20):
                 if zone_x2 > zone_x1 + dx and zone_y2 > zone_y1 + dy:
                     rx = random.uniform(zone_x1 + dx/2, zone_x2 - dx/2)
                     ry = random.uniform(zone_y1 + dy/2, zone_y2 - dy/2)
                     candidates.append((rx, ry))

             best_fallback_z = float('inf')
             best_fallback_state = None

             for fx, fy in candidates:
                 # Calculate gravity support
                 # Add small padding to query dimensions to enforce separation/prevent Z-fighting
                 gravity_z = calculate_z_for_item(fx, fy, dx + 0.001, dy + 0.001, 
                                           placed_bbox[:count], placed_z[:count], placed_h[:count], placed_stackable[:count])
                 
                 if gravity_z > 50000: continue
                 
                 # Try to fit in each layer
                 for layer_z in layer_heights:
                     # Calculate ceiling for this layer
                     layer_ceiling = wh_hgt # Default to max
                     next_layer_exists = False
                     for next_lz in layer_heights:
                        if next_lz > layer_z:
                            layer_ceiling = next_lz
                            next_layer_exists = True
                            break
                     
                     if not next_layer_exists and len(layer_heights) > 1:
                         prev_layer_z = layer_heights[0]
                         for lz in layer_heights:
                             if lz < layer_z: prev_layer_z = lz; break # simplified previous find
                             # Actually logic is: find max lz < layer_z? No, just prev in sorted list
                         # We can just iterate sorted list
                         pass # Skip complex inference in fallback for safety, use wh_hgt or derived logic if easy
                         
                         # Re-use logical derivation if possible, or just default to max or existing ceiling
                         # Let's assume uniform is safer:
                         idx_l = layer_heights.index(layer_z)
                         if idx_l > 0:
                             diff = layer_z - layer_heights[idx_l-1]
                             layer_ceiling = min(wh_hgt, layer_z + diff)
                             
                     z_candidate = max(gravity_z, layer_z)
                     
                     # Enforce ceiling
                     if z_candidate + h <= layer_ceiling + 0.01:
                         # Valid!
                         if z_candidate < best_fallback_z:
                              best_fallback_z = z_candidate
                              best_fallback_state = (fx, fy, z_candidate, fallback_rot, dx, dy)
                              # Break inner layer loop if we found a spot? 
                              # No, we want *best* Z (lowest). Since we sort layers, first valid is lowest for this X,Y?
                              # Gravity might make higher layer valid but lower invalid?
                              # But max(freq, layer) implies increasing Z.
                              # So yes, break to pick lowest valid layer for this X,Y.
                              break
             
             if best_fallback_state:
                 best_state = best_fallback_state
             else:
                 # Absolute Fallback: Find the candidate with the LOWEST forced stack height
                 best_forced_z = float('inf')
                 best_forced_state = None
                 
                 # Backup: just the absolute shortest stack, ignoring layer rules (last resort)
                 shortest_z = float('inf')
                 shortest_state = None

                 for fx, fy in candidates:
                     # Calculate stack height ignoring stackability properties
                     safe_z = calculate_z_for_item(fx, fy, dx + 0.001, dy + 0.001, 
                                               placed_bbox[:count], placed_z[:count], placed_h[:count], placed_stackable[:count],
                                               strict_stacking=False)
                     
                     # Update shortest backup
                     if safe_z < shortest_z:
                         shortest_z = safe_z
                         shortest_state = (fx, fy, safe_z, fallback_rot, dx, dy)

                     # Try to fit into a layer strictly
                     final_z = safe_z
                     valid_layer_found = False
                     
                     for layer_z in layer_heights:
                         # Ceiling logic
                         layer_ceiling = wh_hgt 
                         next_layer_exists = False
                         for next_lz in layer_heights:
                            if next_lz > layer_z:
                                layer_ceiling = next_lz; next_layer_exists = True; break
                         if not next_layer_exists and len(layer_heights) > 1:
                             idx_l = layer_heights.index(layer_z)
                             if idx_l > 0: layer_ceiling = min(wh_hgt, layer_z + (layer_z - layer_heights[idx_l-1]))

                         z_test = max(safe_z, layer_z)
                         
                         # Check strict fit in layer AND warehouse
                         if z_test + items_props[idx, 2] <= layer_ceiling + 0.01 and z_test + items_props[idx, 2] <= wh_hgt + 0.01:
                             final_z = z_test
                             valid_layer_found = True
                             break
                     
                     if valid_layer_found:
                         if final_z < best_forced_z:
                             best_forced_z = final_z
                             best_forced_state = (fx, fy, final_z, fallback_rot, dx, dy)
                 
                 # Decide which fallback to use
                 if best_forced_state:
                     best_state = best_forced_state
                 else:
                     # Even forced stacking failed to find a valid layer spot (warehouse full?)
                     # Use the shortest physical stack found to minimize "Infinite Stacking"
                     if shortest_state:
                         best_state = shortest_state
                     else:
                         # Truly purely hypothetical fallback
                         fx, fy = candidates[0]
                         best_state = (fx, fy, 0.0, fallback_rot, dx, dy)
                 
        # Apply Best State
        b_x, b_y, b_z, b_rot, b_dx, b_dy = best_state
        
        # Handle any remaining penalty Z positions
        if b_z > 50000:
            # Should not happen with strict_stacking=False logic above, but fail safe to MAX height rather than floor
             b_z = 100.0 # Better to be out of bounds than colliding? Or assume 0.0 only if truly no other option?
             # actually if we used strict=False, b_z should be the top of the stack.
             pass
        
        # Ensure Z is never negative
        b_z = max(0.0, b_z)
        
        solution[idx, 0] = b_x
        solution[idx, 1] = b_y
        solution[idx, 2] = b_z
        solution[idx, 3] = b_rot
        
        # Track placed item
        placed_bbox[count] = [b_x - b_dx/2, b_y - b_dy/2, b_x + b_dx/2, b_y + b_dy/2]
        placed_z[count] = b_z
        placed_h[count] = items_props[idx, 2]  # Use original height
        placed_stackable[count] = stackable
        count += 1
        
    return solution


# Helper for layer calculations
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


# --- Standalone Functions for Multiprocessing ---

def create_random_solution_array(num_items, warehouse_dims=None, items_props=None, allocation_zones=None):
    # Use globals if running in worker
    if items_props is None: items_props = _pool_items_props
    if warehouse_dims is None: warehouse_dims = _pool_wh_dims
    if allocation_zones is None: allocation_zones = _pool_allocation_zones
    
    # items_props: (N, 7) array: [length, width, height, can_rotate, stackable, ... ]
    # Only needed cols: 0:len, 1:wid, 2:hgt, 3:can_rot
    # allocation_zones: list of dicts with x1, y1, x2, y2, z1, z2 bounds
    
    solution = np.zeros((num_items, 4), dtype=np.float32)
    
    # Use valid Z positions logic only for fallback or zone limits, but we calculate precise Z now
    
    # Get full warehouse dimensions for proper utilization
    wh_len, wh_wid, wh_hgt = warehouse_dims[:3]
    
    # If allocation zones exist, place items within them
    has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
    
    # Arrays to track placed items for gravity calculation
    # We need to fill solution sequentially
    placed_bboxes = np.zeros((num_items, 4), dtype=np.float32) # x1, y1, x2, y2
    placed_z = np.zeros(num_items, dtype=np.float32)
    placed_h = np.zeros(num_items, dtype=np.float32)
    
    for i in range(num_items):
        item_len = items_props[i, 0]
        item_wid = items_props[i, 1]
        item_hgt = items_props[i, 2]
        can_rotate = items_props[i, 3]
        
        # Retry for floor priority
        best_x, best_y, best_z = 0, 0, float('inf')
        
        # Retry for floor priority (try to find Z=0)
        best_x, best_y, best_z = 0, 0, float('inf')
        
        # Retry for floor priority
        best_x, best_y, best_z = 0, 0, float('inf')
        best_rotation = 0
        
        for attempt in range(50):
            # Dynamic Rotation: Randomize orientation per attempt
            rotation = 0
            if can_rotate and random.random() > 0.5:
                    rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            
            # --- Position Selection Logic ---
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
                # Sort zones by Z height (Bottom Shelf First)
                valid_zones.sort(key=lambda z: z.get('z1', 0))
                
                # Zone Selection Heuristic:
                # Strictly Sequential: 0, 1, 2, ... looping if needed
                # This ensures we try bottom, then next up, then next up...
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
                
                # Dense Pact Heuristic
                if attempt < 5:
                    # Strategy: Bottom-Left Corner Bias
                    x = min_x
                    y = min_y
                elif attempt < 45 and i > 0:
                    # Strategy: Adjacent to existing item
                    rand_idx = random.randint(0, i-1)
                    ref_box = placed_bboxes[rand_idx]
                    # Try to place to the right, or above (in Y)
                    if random.random() < 0.5:
                        x = ref_box[2] + dim_x / 2 # Right
                        y = ref_box[1] + dim_y / 2 # Align Y center
                    else:
                        x = ref_box[0] + dim_x / 2 # Align X center
                        y = ref_box[3] + dim_y / 2 # Top
                    
                    # Perturb slightly to allow sliding
                    x += random.uniform(-1, 1)
                    y += random.uniform(-1, 1)
                    
                    # Clamp
                    x = max(min_x, min(max_x, x))
                    y = max(min_y, min(max_y, y))
                else:
                    # Strategy: Random Fallback
                    x = random.uniform(min_x, max_x)
                    y = random.uniform(min_y, max_y)
                    
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
            
            # Enforce Layer Floor (cannot fall below zone_z1)
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
    accessibility = np.average(access_scores, weights=items_props[:, 5])
    
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
        
        # If rot is 90 or 270, swap
        swap_mask = (rots % 180 != 0)
        current_len = np.where(swap_mask, w, l)
        current_wid = np.where(swap_mask, l, w)
        
        # Half-dims
        hw = current_len / 2
        hh = current_wid / 2
        
        # Z intervals
        z1 = z
        z2 = z + h
        
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
                overlap_x = dx < (hw_batch + hw_other)
                
                # Y overlap
                dy = np.abs(y_batch - y_other)
                overlap_y = dy < (hh_batch + hh_other)
                
                # Z overlap
                # Interval overlap: not (end1 <= start2 or start1 >= end2)
                overlap_z = (z2_batch > (z1_other + 0.01)) & (z1_batch < (z2_other - 0.01))
                
                # Combined
                overlaps = overlap_x & overlap_y & overlap_z
                
                overlap_count += np.sum(overlaps)
        
        # Remove self-overlaps (diagonal was counted once per item)
        # Each item overlaps with itself in the logic above.
        overlap_count -= n
        
        # Divide by 2 because A-B and B-A are counted
        overlap_count /= 2.0
        
        # Normalize
        max_pairs = (n * (n - 1)) / 2 if n > 1 else 1
        overlap_penalty = overlap_count / max_pairs
        
    # --- Stackability Enforcement ---
    # Check if items are stacked on non-stackable items
    stackability_penalty = 0
    # n is already defined
    if n > 1:
        # Reuse variables from earlier or re-extract (safer to re-extract as some were batch-local or overwritten)
        # Actually x,y,z,h were extracted at top of if n > 0 block.
        # But let's be safe and rigorous.
        
        # For each item, check if it's above another item that is NOT stackable
        x = solution[:, 0]
        y = solution[:, 1]
        z = solution[:, 2]
        h = items_props[:, 2]  # heights
        stackable = items_props[:, 4]  # stackable flags
        
        # Get item footprint dimensions (accounting for rotation)
        rots = solution[:, 3].astype(int)
        l = items_props[:, 0]
        w = items_props[:, 1]
        swap_mask = (rots % 180 != 0)
        current_len = np.where(swap_mask, w, l)
        current_wid = np.where(swap_mask, l, w)
        hw = current_len / 2
        hh = current_wid / 2
        
        violations = 0
        for i in range(n):
            if z[i] <= 0.01:  # Item on ground floor, no stacking issue
                continue
            
            # Check all items that could be supporting this item
            for j in range(n):
                if i == j:
                    continue
                
                # Check if item j is below item i
                # Item j's top = z[j] + h[j], Item i's bottom = z[i]
                z_j_top = z[j] + h[j]
                
                # Is item i resting on item j? (within tolerance)
                if abs(z[i] - z_j_top) < 0.1:
                    # Check XY overlap (footprint overlap)
                    dx = abs(x[i] - x[j])
                    dy = abs(y[i] - y[j])
                    overlap_threshold_x = (hw[i] + hw[j]) * 0.5  # At least 50% overlap
                    overlap_threshold_y = (hh[i] + hh[j]) * 0.5
                    
                    if dx < overlap_threshold_x and dy < overlap_threshold_y:
                        # Item i is stacked on item j
                        if stackable[j] < 0.5:  # Item j is NOT stackable
                            violations += 1
                            break  # One violation per stacked item is enough
        
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

    
    norm_weights = {k: v / sum(weights.values()) for k, v in weights.items()}
    
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
        
        cpu_count = multiprocessing.cpu_count()
        # Cap process count to avoid overhead for small tasks or system overload
        process_count = min(cpu_count, 16) 
        
        population = []
        
        if process_count > 1:
             # Prepare args for parallel execution - use None for shared data
             args = [(num_items, None, None, None, None) for _ in range(self.population_size)]
             
             # Use Pool with Initializer
             print(f"Starting parallel initialization with {process_count} processes...")
             # We assume pool is created in optimize() usually, or we create one here.
             # If we create one here, we need to pass the data.
             # BUT: Initialize is called from optimize(), which already sets up the pool context IF we want to reuse it?
             # No, initialize is called once. Optimize re-creates pool or uses one.
             # Better: Create a pool here with initializer.
             
             with multiprocessing.Pool(processes=process_count, initializer=init_worker, 
                                     initargs=(items_props, warehouse_dims, valid_z, allocation_zones, None)) as pool:
                 population = pool.starmap(create_and_repair, args)
        else:
             print("Starting serial initialization...")
             # Set globals for serial execution (shim)
             global _pool_items_props, _pool_wh_dims, _pool_valid_z, _pool_allocation_zones
             old_globals = (_pool_items_props, _pool_wh_dims, _pool_valid_z, _pool_allocation_zones)
             
             _pool_items_props = items_props
             _pool_wh_dims = warehouse_dims
             _pool_valid_z = valid_z
             _pool_allocation_zones = allocation_zones
             
             try:
                 for _ in range(self.population_size):
                     population.append(create_and_repair(num_items, None, None, None, None))
             finally:
                 # Restore
                 _pool_items_props, _pool_wh_dims, _pool_valid_z, _pool_allocation_zones = old_globals

        
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
    
    def fitness_function(self, solution, items, warehouse, weights=None):
        """
        Legacy wrapper for app.py to call the numpy fitness calculation.
        Converts list-based inputs to numpy arrays.
        """
        num_items = len(items)
        
        # Prepare props
        items_props = np.zeros((num_items, 8), dtype=np.float32)
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000 
            ]
            
        # Prepare solution array (N, 4)
        sol_arr = np.zeros((num_items, 4), dtype=np.float32)
        item_map = {item['id']: i for i, item in enumerate(items)}
        
        # Handle solution format (list of dicts vs list of lists?)
        # app.py usually passes list of dicts [{'id':..., 'x':...}, ...]
        if isinstance(solution, list) and len(solution) > 0 and isinstance(solution[0], dict):
             for s in solution:
                 idx = item_map.get(s['id'])
                 if idx is not None:
                     sol_arr[idx] = [s.get('x',0), s.get('y',0), s.get('z',0), s.get('rotation',0)]
        elif isinstance(solution, np.ndarray):
             sol_arr = solution
        
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        
        valid_z = get_valid_z_positions(warehouse)
        zones = get_exclusion_zones(warehouse['id'])
        exclusion_zones_arr = None
        if zones:
             ex_zones = [z for z in zones if z['zone_type'] == 'exclusion']
             if ex_zones:
                 exclusion_zones_arr = np.array([[z['x1'], z['y1'], z['x2'], z['y2']] for z in ex_zones])
                 
        return fitness_function_numpy(sol_arr, items_props, wh_dims, weights, valid_z, exclusion_zones_arr)


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
            
        # Pre-process items into numpy array (N, 8)
        # Cols: 0:len, 1:wid, 2:hgt, 3:can_rot, 4:stackable, 5:access_freq, 6:weight, 7:category_hash
        items_props = np.zeros((num_items, 8), dtype=np.float32)
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000 
            ]
            
        wh_dims = (warehouse['length'], warehouse['width'], warehouse['height'], 
                   warehouse.get('door_x', 0), warehouse.get('door_y', 0))
        
        # DEBUG: Log unstackable count
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
        
        cpu_count = multiprocessing.cpu_count()
        process_count = min(cpu_count, 16)
        
        # Create pool with shared memory initializer
        with multiprocessing.Pool(processes=process_count, initializer=init_worker, 
                                initargs=(items_props, wh_dims, valid_z, allocation_zones, exclusion_zones_arr)) as pool:
            
            for generation in range(self.generations):
                if optimization_state and not optimization_state['running']:
                    break
                    
                fitness_scores = []
                metrics_list = []
                
                if callback:
                     msg = f"GA Generation {generation + 1}/{self.generations}"
                     # Send immediate status update
                     callback((generation) / self.generations * 100, best_fitness if best_fitness != -float('inf') else 0, best_fitness if best_fitness != -float('inf') else 0, None, 0, 0, 0, message=msg)

                
                current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
                
                # Parallel Fitness Evaluation
                if process_count > 1:
                     # args: (solution, None, None, weights, None, None) -> using globals
                     fit_args = [(sol, None, None, current_weights, None, None) for sol in population]
                     results = pool.starmap(fitness_function_numpy, fit_args)
                     
                     # results is list of tuples (f, su, acc, sta, grp)
                     results_arr = np.array(results)
                     fitness_scores = results_arr[:, 0]
                     metrics_list = results_arr[:, 1:]
                else:
                    # Shim globals for serial fallback
                    global _pool_items_props, _pool_wh_dims, _pool_valid_z, _pool_allocation_zones, _pool_exclusion_zones
                    _pool_items_props = items_props
                    _pool_wh_dims = wh_dims
                    _pool_valid_z = valid_z
                    _pool_allocation_zones = allocation_zones
                    _pool_exclusion_zones = exclusion_zones_arr
                    
                    for sol in population:
                        f, su, acc, sta, grp = fitness_function_numpy(
                            sol, None, None, current_weights, None, None)
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
                     # Prepare batches
                     offspring_args = []
                     for i in range(0, len(selected_pop), 2):
                         if i+1 < len(selected_pop):
                              p1 = selected_pop[i]
                              p2 = selected_pop[i+1]
                              # Pass None for shared data
                              offspring_args.append((p1, p2, self.crossover_rate, self.mutation_rate, None, None, None, None))
                         else:
                              offspring_args.append((selected_pop[i], selected_pop[i], self.crossover_rate, self.mutation_rate, None, None, None, None))
                     
                     if offspring_args:
                         results = pool.starmap(process_offspring_batch, offspring_args)
                         for pair in results:
                             new_pop.extend(pair)
                else:
                    # Serial Fallback (globals already shimmed above)
                    for i in range(0, len(selected_pop), 2):
                        if i+1 < len(selected_pop):
                             batch_res = process_offspring_batch(selected_pop[i], selected_pop[i+1], self.crossover_rate, self.mutation_rate, None, None, None, None)
                             new_pop.extend(batch_res)
                        else:
                             batch_res = process_offspring_batch(selected_pop[i], selected_pop[i], self.crossover_rate, self.mutation_rate, None, None, None, None)
                             new_pop.extend(batch_res)
                
                new_pop = new_pop[:self.population_size]
                population = np.array(new_pop)
                
                # Callback
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
                
        items_props = np.zeros((num_items, 8))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000 
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
        items_props = np.zeros((num_items, 8))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000 
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
        
        for iteration in range(self.iterations):
            if optimization_state and not optimization_state['running']:
                break
                
            # Calculate individual item fitnesses (local contribution)
            item_fitnesses = np.zeros(num_items)
            
            # Simple heuristic: item fitness based on distance to door and zone collision
            door_x = wh_dims[3] if len(wh_dims) >= 5 else 0
            door_y = wh_dims[4] if len(wh_dims) >= 5 else 0
            
            for i in range(num_items):
                dist = np.sqrt((solution[i, 0] - door_x)**2 + (solution[i, 1] - door_y)**2)
                access_score = 1.0 / (1.0 + dist) * items_props[i, 5]  # Weighted by access_freq
                item_fitnesses[i] = access_score
            
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
            
            # Mutate the selected item (try new random position)
            old_pos = solution[item_to_mutate].copy()
            
            item_len = items_props[item_to_mutate, 0]
            item_wid = items_props[item_to_mutate, 1]
            can_rotate = items_props[item_to_mutate, 3]
            
            rotation = old_pos[3]
            if can_rotate and random.random() > 0.7:
                rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            
            wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
            
            # --- Gravity Calculation Prep ---
            # Pre-calculate other items' bboxes ONCE
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
            
            # Corrected Mutation Logic:
            # 1. Randomize position or pick a random valid zone to seed.
            # 2. Let repair_solution_compact do the heavy lifting.
            
            has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
            
            seed_x, seed_y, seed_z = 0, 0, 0
            
            if has_allocation_zones:
                 # Pick a random zone to start in
                 seed_zone = random.choice(allocation_zones)
                 
                 # Random pos in that zone
                 z_min_x = seed_zone['x1']
                 z_max_x = seed_zone['x2']
                 z_min_y = seed_zone['y1']
                 z_max_y = seed_zone['y2']
                 
                 # Adjust for item size
                 item_h = items_props[item_to_mutate, 2]
                 if int(rotation) % 180 == 0:
                     dx, dy = item_len, item_wid
                 else:
                     dx, dy = item_wid, item_len
                 
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
                 # Full warehouse random
                 seed_x = random.uniform(0, wh_len)
                 seed_y = random.uniform(0, wh_wid)
                 seed_z = 0
            
            # Apply seed
            solution[item_to_mutate] = [seed_x, seed_y, seed_z, rotation]
            
            # Apply gravity repair to ensure no floating items and strict zone adherence
            # This calls the SAME function we fixed for GA, so it iterates ALL zones and layers.
            solution = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, valid_z)
            new_fitness, su, acc, sta, grp = fitness_function_numpy(
                solution=solution, items_props=items_props, warehouse_dims=wh_dims, 
                weights=current_weights, valid_z=valid_z, exclusion_zones_arr=exclusion_zones_arr)

            
            # Accept if better, otherwise revert
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_solution = solution.copy()
                time_to_best = time.time() - start_time
            else:
                # Revert mutation
                solution[item_to_mutate] = old_pos
        
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
    def __init__(self, ga_generations=500, eo_iterations=1000):
        self.ga_generations = ga_generations
        self.eo_iterations = eo_iterations
        
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


        
        ga = GeneticAlgorithm(generations=self.ga_generations)
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
        
        items_props = np.zeros((num_items, 8))
        for i, item in enumerate(items):
            items_props[i] = [
                item['length'], item['width'], item['height'],
                item['can_rotate'], item['stackable'],
                item['access_freq'], item.get('weight', 0),
                hash(item.get('category', '')) % 10000 
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
        
        # EO refinement phase
        for iteration in range(self.eo_iterations):
            if optimization_state and not optimization_state['running']:
                break
            
            # Calculate individual item fitnesses
            item_fitnesses = np.zeros(num_items)
            door_x = wh_dims[3] if len(wh_dims) >= 5 else 0
            door_y = wh_dims[4] if len(wh_dims) >= 5 else 0
            
            for i in range(num_items):
                dist = np.sqrt((solution[i, 0] - door_x)**2 + (solution[i, 1] - door_y)**2)
                access_score = 1.0 / (1.0 + dist) * items_props[i, 5]
                item_fitnesses[i] = access_score
            
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
            
            rotation = old_pos[3]
            if can_rotate and random.random() > 0.7:
                rotation = random.choice([0, 90, 180, 270])
            
            if int(rotation) % 180 == 0:
                dim_x, dim_y = item_len, item_wid
            else:
                dim_x, dim_y = item_wid, item_len
            
            wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
            
            # --- Gravity Calculation Prep ---
            # Pre-calculate other items' bboxes ONCE
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
            
            for attempt in range(50):
                # Check if we should use allocation zones
                has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
                
                zone_z1 = 0
                zone_z2 = wh_hgt
                if has_allocation_zones:
                    # Find zones that can fit this item
                    item_hgt = items_props[item_to_mutate, 2]
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
                    best_x, best_y, best_z = x, y, z
                
                if z == zone_z1:
                    break
            
            if best_z > 50000:
                best_z -= 100000.0
            
            x, y, z = best_x, best_y, best_z
            solution[item_to_mutate] = [x, y, z, rotation]
            
            # Apply gravity repair to ensure no floating items
            solution = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, valid_z)
            
            new_fitness, su, acc, sta, grp = fitness_function_numpy(
                solution=solution, items_props=items_props, warehouse_dims=wh_dims, 
                weights=current_weights, valid_z=valid_z, exclusion_zones_arr=exclusion_zones_arr)
            
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_solution = solution.copy()
                time_to_best = time.time() - start_time
            else:
                solution[item_to_mutate] = old_pos
            
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

        
        ga = GeneticAlgorithm(generations=self.ga_generations)
        final_solution, final_fitness, final_time = ga.optimize(
             items, warehouse, weights, ga_callback, optimization_state, initial_solution=eo_solution_list
        )
        
        total_time = time.time() - start_time
        return final_solution, final_fitness, total_time
