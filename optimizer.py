import math
import random
import numpy as np
import time
from database import get_exclusion_zones

# Helper for gravity calculation
def calculate_z_for_item(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h):
    """
    Calculate the lowest valid Z position for an item given its position and other items.
    
    Args:
        x, y: Center coordinates of the new item
        dim_x, dim_y: Dimensions of the new item
        other_items_bbox: (N, 4) array of [min_x, min_y, max_x, max_y] for placed items
        other_items_z: (N,) array of Z positions for placed items
        other_items_h: (N,) array of heights for placed items
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


def repair_solution_compact(solution, items_props, warehouse_dims=None, allocation_zones=None, layer_heights=None):
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
    
    # Sort indices by volume (larger items first for better packing)
    volumes = items_props[:, 0] * items_props[:, 1] * items_props[:, 2]
    sorted_indices = np.argsort(-volumes)  # Descending by volume
    
    # Arrays to track placed items
    placed_bbox = np.zeros((num_items, 4))  # x1, y1, x2, y2
    placed_z = np.zeros(num_items)
    placed_h = np.zeros(num_items)
    
    count = 0
    
    for idx in sorted_indices:
        l, w, h = items_props[idx, 0:3]
        can_rotate = items_props[idx, 3]
        
        # Try all rotation options if rotation is allowed (0, 90 degrees)
        orientations_to_try = [0.0, 90.0] if can_rotate else [solution[idx, 3]]
        
        best_state = None
        best_score = float('inf')
        
        # Get zone bounds for this item (use first valid zone or full warehouse)
        zone_x1, zone_y1, zone_z1 = 0.0, 0.0, 0.0
        zone_x2, zone_y2, zone_z2 = wh_len, wh_wid, wh_hgt
        
        if allocation_zones and len(allocation_zones) > 0:
            # Try to find a zone that fits this item
            for zone in allocation_zones:
                zw = zone['x2'] - zone['x1']
                zh = zone['y2'] - zone['y1']
                if min(l, w) <= max(zw, zh) and max(l, w) <= max(zw, zh):
                    zone_x1 = zone['x1']
                    zone_y1 = zone['y1']
                    zone_x2 = zone['x2']
                    zone_y2 = zone['y2']
                    zone_z1 = zone.get('z1', 0.0)
                    zone_z2 = zone.get('z2', wh_hgt)
                    break
        
        for rot in orientations_to_try:
            # Calculate Dimensions based on rotation
            if int(rot) % 180 == 0:
                dx, dy = l, w
            else:
                dx, dy = w, l
            
            # Skip if item doesn't fit in zone at all
            if dx > (zone_x2 - zone_x1) or dy > (zone_y2 - zone_y1):
                continue
            
            # Try each layer, starting from bottom
            for layer_z in layer_heights:
                if layer_z < zone_z1 or layer_z >= zone_z2:
                    continue
                
                # Calculate layer ceiling (next layer height or zone top)
                layer_ceiling = zone_z2
                for next_lz in layer_heights:
                    if next_lz > layer_z:
                        layer_ceiling = next_lz
                        break
                
                # Check if item fits within layer height
                if layer_z + h > layer_ceiling + 0.01:  # Small tolerance
                    continue  # Item too tall for this layer, try next layer
                
                # --- Tetris-like placement: scan for first open position ---
                # Strategy: Try positions in a grid pattern from zone origin
                step_x = max(0.5, dx / 2)
                step_y = max(0.5, dy / 2)
                
                # Start from zone corner
                test_x = zone_x1 + dx / 2
                found_position = False
                
                while test_x + dx / 2 <= zone_x2 + 0.01:
                    test_y = zone_y1 + dy / 2
                    
                    while test_y + dy / 2 <= zone_y2 + 0.01:
                        # Check Z at this position
                        z = calculate_z_for_item(test_x, test_y, dx, dy, 
                                                 placed_bbox[:count], placed_z[:count], placed_h[:count])
                        
                        # Apply layer floor constraint
                        z = max(z, layer_z)
                        
                        # Check if item fits within layer ceiling
                        if z + h <= layer_ceiling + 0.01:
                            # Valid position found!
                            # Calculate score: prefer lower Z, then closer to origin
                            score = z * 10000.0 + test_x + test_y
                            
                            if score < best_score:
                                best_score = score
                                best_state = (test_x, test_y, z, rot, dx, dy)
                                found_position = True
                        
                        test_y += step_y
                    
                    test_x += step_x
                
                # If we found a valid position on this layer, no need to check higher layers
                if found_position:
                    break
        
        # If no valid position found, use fallback (original position clamped to zone)
        if best_state is None:
            fallback_rot = solution[idx, 3]
            if int(fallback_rot) % 180 == 0:
                dx, dy = l, w
            else:
                dx, dy = w, l
            
            fallback_x = max(zone_x1 + dx/2, min(zone_x2 - dx/2, solution[idx, 0]))
            fallback_y = max(zone_y1 + dy/2, min(zone_y2 - dy/2, solution[idx, 1]))
            fallback_z = max(zone_z1, min(zone_z2 - h, solution[idx, 2]))
            best_state = (fallback_x, fallback_y, fallback_z, fallback_rot, dx, dy)
        
        # Apply Best State
        b_x, b_y, b_z, b_rot, b_dx, b_dy = best_state
        
        # Handle any remaining penalty Z positions
        if b_z > 50000:
            b_z -= 100000.0
            b_z = max(0.0, b_z)
        
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
        population = []
        for _ in range(self.population_size):
             sol = self.create_random_solution_array(num_items, warehouse_dims, items_props, allocation_zones)
             sol = repair_solution_compact(sol, items_props, warehouse_dims, allocation_zones, valid_z)
             population.append(sol)
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

    def create_random_solution_array(self, num_items, warehouse_dims, items_props, allocation_zones=None):
        # items_props: (N, 7) array: [length, width, height, can_rotate, stackable, ... ]
        # Only needed cols: 0:len, 1:wid, 2:hgt, 3:can_rot
        # allocation_zones: list of dicts with x1, y1, x2, y2, z1, z2 bounds
        
        solution = np.zeros((num_items, 4))
        
        # Use valid Z positions logic only for fallback or zone limits, but we calculate precise Z now
        
        # Get full warehouse dimensions for proper utilization
        wh_len, wh_wid, wh_hgt = warehouse_dims[:3]
        
        # If allocation zones exist, place items within them
        has_allocation_zones = allocation_zones is not None and len(allocation_zones) > 0
        
        # Arrays to track placed items for gravity calculation
        # We need to fill solution sequentially
        placed_bboxes = np.zeros((num_items, 4)) # x1, y1, x2, y2
        placed_z = np.zeros(num_items)
        placed_h = np.zeros(num_items)
        
        for i in range(num_items):
            item_len = items_props[i, 0]
            item_wid = items_props[i, 1]
            item_hgt = items_props[i, 2]
            can_rotate = items_props[i, 3]
            
            # Retry for floor priority
            best_x, best_y, best_z = 0, 0, float('inf')
            
            # Retry logic for floor priority (try to find Z=0)
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

    # Optimized collision check NOT used in full batch init for speed,
    # but used during mutation/repair
    
    def fitness_function_numpy(self, solution, items_props, warehouse_dims, weights, valid_z, exclusion_zones_arr=None):
        # solution: (N, 4)
        # items_props: (N, 8) cols: len, wid, hgt, can_rot, stackable, access_freq, weight, category_id
        # exclusion_zones_arr: (K, 4) -> x1, y1, x2, y2
        
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
        for z_pos in valid_z:
            on_valid_z |= (np.abs(solution[:, 2] - z_pos) < 0.001)
        stability = np.mean(on_valid_z)
        
        # Exclusion Zones
        # solution x, y are centers. We need to check if bounding box overlaps zone.
        # For speed, check center point first? Or approximation?
        # Let's check if center is inside zone.
        # zones: x1, y1, x2, y2
        zone_penalty = 0
        if exclusion_zones_arr is not None and len(exclusion_zones_arr) > 0:
            # Broadcast check: (N, 1, 2) vs (1, K, 4) ?
            # Sol x,y: (N, 1)
            # Zone x1, x2: (1, K)
            
            x = solution[:, 0:1] # (N, 1)
            y = solution[:, 1:2] # (N, 1)
            
            z_x1 = exclusion_zones_arr[:, 0] # (K,)
            z_y1 = exclusion_zones_arr[:, 1]
            z_x2 = exclusion_zones_arr[:, 2]
            z_y2 = exclusion_zones_arr[:, 3]
            
            # Simple check: is center inside?
            # inside_x = (x >= z_x1) & (x <= z_x2)
            # inside_y = (y >= z_y1) & (y <= z_y2)
            # collisions = inside_x & inside_y # (N, K)
            # any_collision = np.any(collisions, axis=1) # (N,)
            
            # Better: AABB overlap
            # Item dims (approximation with non-rotated len/wid for speed or use max dim)
            # Let's use max dim / 2 for radius
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
            
            # Overlap if dx < (radius + zone_half_width)
            # This treats items as circles roughly or boxes. 
            # Correct AABB vs AABB: 
            # overlap_x = dx < (item_half_w + zone_half_w)
            
            # Vectorized AABB with rotation ignored (using max dim covers worst case)
            collision_x = dx < (radii + z_hw)
            collision_y = dy < (radii + z_hh)
            collisions = collision_x & collision_y
            
            zone_penalty = np.sum(collisions) / len(solution) # Fraction of items colliding
            
        # --- Item-Item Overlap ---
        # Vectorized O(N^2) check. For large N, this needs spatial indexing, but for < ~2000 it's fast on GPU/CPU with numpy
        n = len(solution)
        if n > 0:
             # Extract centers (n, 1) and (1, n)
             x = solution[:, 0]
             y = solution[:, 1]
             # z = solution[:, 2] # Ignore Z for now per user request context (implied 2D layout focus) or check 3D?
             # User said "overlap", usually implies 2D footprint unless stacking is key.
             # Given "stability" check handles Z-levels, let's strictly enforce 2D non-overlap for same Z?
             # Actually, items can be stacked? 
             # If stacking is allowed, we only penalize if Z ranges overlap too.
             # For now, simplistic 2D footprint overlap penalty is stricter and safer.
             
             # Widths/Lengths (halved for radius-like check or full AABB)
             # Account for rotation in dims
             # items_props: 0:len, 1:wid
             # solution: 3:rot
             
             # We need actual dims based on rotation per item
             # 0 deg: len=l, wid=w. 90 deg: len=w, wid=l
             # Vectorize dim swapping where rot % 180 != 0
             
             rots = solution[:, 3].astype(int)
             l = items_props[:, 0]
             w = items_props[:, 1]
             
             # If rot is 90 or 270, swap
             swap_mask = (rots % 180 != 0)
             current_len = np.where(swap_mask, w, l)
             current_wid = np.where(swap_mask, l, w)
             
             # AABB Half-dims
             hw = current_len / 2
             hh = current_wid / 2
             
             # Pairwise distance (N, N)
             x_matrix = x.reshape(-1, 1) # col
             x_matrix_T = x.reshape(1, -1) # row
             dx = np.abs(x_matrix - x_matrix_T)
             
             y_matrix = y.reshape(-1, 1)
             y_matrix_T = y.reshape(1, -1) # row
             dy = np.abs(y_matrix - y_matrix_T)
             
             # Pairwise sum of half-widths
             hw_matrix = hw.reshape(-1, 1) + hw.reshape(1, -1)
             hh_matrix = hh.reshape(-1, 1) + hh.reshape(1, -1)
             
             # Overlap conditions
             overlap_x = dx < hw_matrix
             overlap_y = dy < hh_matrix
             overlaps = overlap_x & overlap_y
             
             # Remove self-overlaps (diagonal)
             np.fill_diagonal(overlaps, False)
             
             # Check Z overlap if 3D?
             # For improved realism:
             # z_bottom = solution[:, 2]
             # z_top = z_bottom + items_props[:, 2]
             # dz_overlap ...
             # For now, let's assume strict 2D separation required or strict 3D.
             # Let's add 3D check to allow valid stacking
             z = solution[:, 2]
             h = items_props[:, 2]
             z_matrix = z.reshape(-1, 1)
             z_matrix_T = z.reshape(1, -1)
             dz_centers = np.abs(z_matrix - z_matrix_T) # This is center dist? No z is bottom.
             
             # Interval overlap: [z1, z1+h1] vs [z2, z2+h2]
             # Overlap if z1 < z2+h2 AND z2 < z1+h1
             z1 = z_matrix
             h1 = h.reshape(-1, 1)
             z2 = z_matrix_T
             h2 = h.reshape(1, -1)
             
             overlap_z = (z1 < (z2 + h2 - 0.01)) & (z2 < (z1 + h1 - 0.01)) # Epsilon for touching
             
             overlaps = overlaps & overlap_z
             
             overlap_count = np.sum(overlaps) / 2 # Divide by 2 because symmetric (A overlaps B, B overlaps A)
             
             # Penalty proportional to number of overlapping pairs
             # Normalize by max possible pairs N*(N-1)/2
             max_pairs = (n * (n - 1)) / 2 if n > 1 else 1
             overlap_penalty = overlap_count / max_pairs
             
        # --- Stackability Enforcement ---
        # Check if items are stacked on non-stackable items
        stackability_penalty = 0
        n = len(solution)
        if n > 1:
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
        # Calculate how close items of the same category are to each other.
        # Minimal distance to category centroid.
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
                # Normalize: 1 / (1 + avg_dist / constant)
                # If avg_dist is small, grouping is high.
                # Use a scaling factor?
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
            fitness -= (overlap_penalty * 2.0) # Heavy penalty (can go negative) to force separation
            # Or multiply: fitness *= (1.0 - overlap_penalty) ?
            # User wants "make sure NO items overlap". 
            # Multiplication is safer to keep bounds, but subtraction is stronger gradient.
            # Let's use severe multiplication
            fitness *= max(0, (1.0 - overlap_penalty * 5.0)) # 20% overlap = 0 fitness

        # Apply stackability penalty - items on non-stackable items
        if stackability_penalty > 0:
            fitness *= max(0, (1.0 - stackability_penalty * 3.0))

        # Prefer lower Z (floor usage) if possible - reducing fitness slightly as average Z increases
        avg_z = np.mean(solution[:, 2])
        wh_hgt_val = warehouse_dims[2]
        if wh_hgt_val > 0:
            fitness *= (1.0 - (avg_z / wh_hgt_val) * 0.15) 

        return fitness, space_util, accessibility, stability, grouping

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

    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        if len(parent1) < 2:
            return parent1.copy(), parent2.copy()

        # Single point crossover along the item axis
        point = random.randint(1, len(parent1) - 1)
        child1 = np.vstack((parent1[:point], parent2[point:]))
        child2 = np.vstack((parent2[:point], parent1[point:]))
        return child1, child2
        
    def mutation(self, solution, warehouse_dims, items_props, valid_z, allocation_zones=None):
        # Randomly mutate 1 item
        if random.random() < self.mutation_rate:
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
                
                z = calculate_z_for_item(x, y, dim_x, dim_y, other_bbox, other_z, other_h)
                
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
        
        population = self.initialize_population(num_items, wh_dims, items_props, valid_z, allocation_zones)
        
        if initial_solution is not None:
             # Convert to numpy (N, 4)
             sol_arr = np.zeros((num_items, 4))
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
        
        for generation in range(self.generations):
            if optimization_state and not optimization_state['running']:
                break
                
            fitness_scores = []
            metrics_list = []
            
            # Vectorize fitness calc if possible or loop
            # Python loop over pop (size 50) is fine.
            # Use default weights if None
            current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
            for sol in population:
                f, su, acc, sta, grp = self.fitness_function_numpy(
                    sol, items_props, wh_dims, current_weights, valid_z, exclusion_zones_arr)
                fitness_scores.append(f)
                metrics_list.append((su, acc, sta, grp))
                
            fitness_scores = np.array(fitness_scores)
            best_idx = np.argmax(fitness_scores)
            
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_solution = population[best_idx].copy()
                time_to_best = time.time() - start_time
                
            # Selection
            selected_pop = self.selection(population, fitness_scores)
            
            # Crossover & Mutation
            new_pop = []
            for i in range(0, len(selected_pop), 2):
                if i+1 < len(selected_pop):
                    c1, c2 = self.crossover(selected_pop[i], selected_pop[i+1])
                    m1 = self.mutation(c1, wh_dims, items_props, valid_z, allocation_zones)
                    m2 = self.mutation(c2, wh_dims, items_props, valid_z, allocation_zones)
                    
                    # Apply gravity repair to ensure no floating items
                    m1 = repair_solution_compact(m1, items_props, wh_dims, allocation_zones, valid_z)
                    m2 = repair_solution_compact(m2, items_props, wh_dims, allocation_zones, valid_z)
                    
                    new_pop.append(m1)
                    new_pop.append(m2)
                else:
                    m = self.mutation(selected_pop[i], wh_dims, items_props, valid_z, allocation_zones)
                    m = repair_solution_compact(m, items_props, wh_dims, allocation_zones, valid_z)
                    new_pop.append(m)
            
            population = np.array(new_pop)
            
            # Callback
            if callback:
                progress = (generation + 1) / self.generations * 100
                avg_fitness = np.mean(fitness_scores)
                # Just use mean of metrics from this gen
                avg_space = np.mean([m[0] for m in metrics_list])
                avg_acc = np.mean([m[1] for m in metrics_list])
                avg_stab = np.mean([m[2] for m in metrics_list])
                
                # Convert best_solution to list of dicts for visualization
                # For large datasets (400k items), this is expensive, so we throttle it.
                # Update visual every 5 generations or on the last generation.
                converted_solution = None
                if best_solution is not None and (generation % 5 == 0 or generation == self.generations - 1):
                    converted_solution = []
                    # Vectorized conversion if possible? 
                    # We need IDs from items list.
                    # Fast loop:
                    for i in range(num_items):
                        converted_solution.append({
                            'id': items[i]['id'],
                            'x': float(best_solution[i, 0]),
                            'y': float(best_solution[i, 1]),
                            'z': float(best_solution[i, 2]),
                            'rotation': int(best_solution[i, 3])
                        })
                
                callback(progress, avg_fitness, best_fitness, converted_solution,
                         avg_space, avg_acc, avg_stab)

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

        return self.fitness_function_numpy(sol_array, items_props, wh_dims, 
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
        ga_helper = GeneticAlgorithm()
        solution = ga_helper.create_random_solution_array(num_items, wh_dims, items_props, allocation_zones)
        solution = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, valid_z)
        
        best_solution = solution.copy()
        current_weights = weights if weights else {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1}
        best_fitness, _, _, _, _ = ga_helper.fitness_function_numpy(
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
            
            for attempt in range(50):
                # Dynamic Rotation
                rotation = solution[item_to_mutate, 3]
                can_rotate = items_props[item_to_mutate, 3]
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
                
                z = calculate_z_for_item(x, y, dim_x, dim_y, other_bbox, other_z, other_h)
                
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
            solution = repair_solution_compact(solution, items_props)
            
            # Evaluate new solution
            new_fitness, su, acc, sta, grp = ga_helper.fitness_function_numpy(
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
                if iteration % 50 == 0 or iteration == self.iterations - 1:
                    converted_solution = []
                    for i in range(num_items):
                        converted_solution.append({
                            'id': items[i]['id'],
                            'x': float(best_solution[i, 0]),
                            'y': float(best_solution[i, 1]),
                            'z': float(best_solution[i, 2]),
                            'rotation': int(best_solution[i, 3])
                        })
                
                callback(progress, new_fitness, best_fitness, converted_solution, su, acc, sta)
        
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
        def ga_callback(progress, avg_fit, best_fit, solution, space, access, stability):
            if callback:
                # Scale GA progress to 0-70%
                callback(progress * 0.7, avg_fit, best_fit, solution, space, access, stability)
        
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
                
                z = calculate_z_for_item(x, y, dim_x, dim_y, other_bbox, other_z, other_h)
                
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
            
            new_fitness, su, acc, sta, grp = ga_helper.fitness_function_numpy(
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
                if iteration % 50 == 0 or iteration == self.eo_iterations - 1:
                    converted_solution = []
                    for i in range(num_items):
                        converted_solution.append({
                            'id': items[i]['id'],
                            'x': float(best_solution[i, 0]),
                            'y': float(best_solution[i, 1]),
                            'z': float(best_solution[i, 2]),
                            'rotation': int(best_solution[i, 3])
                        })
                
                callback(total_progress, new_fitness, best_fitness, converted_solution, su, acc, sta)
        
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
        def eo_callback(progress, avg_fit, best_fit, solution, space, access, stability):
            if callback:
                # Scale EO progress to 0-70%
                callback(progress * 0.7, avg_fit, best_fit, solution, space, access, stability)
                
        eo = ExtremalOptimization(iterations=self.eo_iterations)
        eo_solution_list, eo_fitness, eo_time = eo.optimize(
            items, warehouse, weights, eo_callback, optimization_state
        )
        
        if optimization_state and not optimization_state['running']:
            return eo_solution_list, eo_fitness, eo_time
            
        # Phase 2: Run GA for refinement using EO solution as seed (30% of progress)
        def ga_callback(progress, avg_fit, best_fit, solution, space, access, stability):
            if callback:
                # Scale GA progress to 70-100%
                total = 70 + (progress * 0.3)
                callback(total, avg_fit, best_fit, solution, space, access, stability)
        
        ga = GeneticAlgorithm(generations=self.ga_generations)
        final_solution, final_fitness, final_time = ga.optimize(
             items, warehouse, weights, ga_callback, optimization_state, initial_solution=eo_solution_list
        )
        
        total_time = time.time() - start_time
        return final_solution, final_fitness, total_time
