import numpy as np

def calculate_z_for_item_debug(x, y, dim_x, dim_y, other_items_bbox, other_items_z, other_items_h, other_items_stackable=None):
    # Dimensions of new item
    new_min_x = x - dim_x / 2
    new_max_x = x + dim_x / 2
    new_min_y = y - dim_y / 2
    new_max_y = y + dim_y / 2
    
    # Check for overlaps with placed items
    # (Checking intersection of rectangles)
    # Overlap X: start1 < end2 AND start2 < end1
    overlaps_x = (new_min_x < other_items_bbox[:, 2]) & (new_max_x > other_items_bbox[:, 0])
    overlaps_y = (new_min_y < other_items_bbox[:, 3]) & (new_max_y > other_items_bbox[:, 1])
    overlaps = overlaps_x & overlaps_y
    
    # If no overlaps, z is 0 (or floor)
    if not np.any(overlaps):
        return 0.0
        
    # Check if any overlapping item is NOT stackable
    if other_items_stackable is not None:
        overlapping_stackables = other_items_stackable[overlaps]
        print(f"DEBUG: Overlapping stackables: {overlapping_stackables}")
        if np.any(overlapping_stackables == 0):
             return 1000000.0 # Effectively impossible
             
    # Calculate max Z of overlapping items
    overlapping_z_tops = other_items_z[overlaps] + other_items_h[overlaps]
    max_z = np.max(overlapping_z_tops)
    
    return max_z

def test_stack_logic():
    print("Testing Stack Logic...")
    
    # Scenario: Item 0 is Unstackable (stackable=0), placed at (0,0) size 10x10, z=0, h=10
    placed_bbox = np.array([[ -5, -5, 5, 5 ]]) # min_x, min_y, max_x, max_y (centered at 0,0, w=10, l=10)
    placed_z = np.array([0.0])
    placed_h = np.array([10.0])
    placed_stackable = np.array([0], dtype=int)
    
    # Try to place Item 1 (Stackable) on top at (0,0) size 10x10
    # Overlap should be detected
    z = calculate_z_for_item_debug(0, 0, 10, 10, placed_bbox, placed_z, placed_h, placed_stackable)
    
    print(f"Result Z: {z}")
    if z >= 1000000.0:
        print("PASS: Blocked")
    else:
        print("FAIL: Not Blocked")

    # Scenario 2: Item 0 is Stackable (stackable=1)
    placed_stackable_2 = np.array([1], dtype=int)
    z2 = calculate_z_for_item_debug(0, 0, 10, 10, placed_bbox, placed_z, placed_h, placed_stackable_2)
    print(f"Result Z2: {z2}")
    if z2 == 10.0:
        print("PASS: Stacked correctly")
    else:
        print(f"FAIL: Expected 10.0, got {z2}")

if __name__ == "__main__":
    test_stack_logic()
