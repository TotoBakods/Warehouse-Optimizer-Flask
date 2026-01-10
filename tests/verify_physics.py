import pybullet as p
import pybullet_data
import time
import numpy as np
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizer import repair_solution_compact

def run_physics_verification():
    print("--- PyBullet Physics Verification ---")
    
    # 1. Setup Optimizer Scenario
    wh_dims = (100, 100, 100, 0, 0) # L, W, H
    layer_heights = [0.0]
    allocation_zones = [{'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100, 'z1': 0, 'z2': 100}]
    
    # Generate 50 items
    num_items = 50
    items_props = np.zeros((num_items, 8), dtype=np.float32)
    for i in range(num_items):
        l = random.uniform(5, 15)
        w = random.uniform(5, 15)
        h = random.uniform(5, 15)
        items_props[i] = [l, w, h, 0, 1, 0, 10, 0] # non-rotatable for simplicity? Let's say rotatable=0
    
    # Run Optimizer
    print("Running Optimizer...")
    solution = np.zeros((num_items, 4))
    optimized_sol = repair_solution_compact(solution, items_props, wh_dims, allocation_zones, layer_heights)
    
    # 2. Setup PyBullet
    print("Initializing Physics Engine...")
    # Use DIRECT mode for speed (no GUI), or GUI to see it
    # p.connect(p.GUI) 
    p.connect(p.DIRECT) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Create Floor
    planeId = p.loadURDF("plane.urdf")
    
    # Create Items
    collision_count = 0
    item_ids = []
    
    print("Spawning Items in Physics World...")
    for i in range(num_items):
        item_x = optimized_sol[i, 0]
        item_y = optimized_sol[i, 1]
        item_z = optimized_sol[i, 2]
        item_rot = optimized_sol[i, 3]
        
        l, w, h = items_props[i, 0:3]
        
        # Handle Rotation swapping dims
        if int(item_rot) % 180 == 0:
            dx, dy = l, w
        else:
            dx, dy = w, l
            
        # PyBullet Shapes (half-extents)
        colShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dx/2.0, dy/2.0, h/2.0])
        visualShapeId = -1 # No visual needed for headless
        
        # Optimizer Coords are Center? Or Bottom-Left-Corner?
        # Based on my `calculate_z_for_item`, coordinates passed to optimizer are typically expected to be...
        # Wait, `optimizer.py` line 33 says: "x, y: Center coordinates".
        # But `repair_solution_compact` output:
        # line 479: solution[idx, 0] = b_x  <-- best_state was (fx, fy...) where fx is CENTER.
        # So output is CENTER.
        # PyBullet expects Center of Mass.
        # Typically Center of Mass for a box is its Geometric Center.
        # But `z` in optimizer usually means "Bottom Z".
        # PyBullet Z is Center Z.
        
        pybullet_z = item_z + h/2.0
        
        # Create Body
        basePosition = [item_x, item_y, pybullet_z]
        baseOrientation = p.getQuaternionFromEuler([0, 0, 0]) # Rotation already handled by swapping dims
        
        bodyId = p.createMultiBody(baseMass=1, 
                                   baseCollisionShapeIndex=colShapeId, 
                                   baseVisualShapeIndex=visualShapeId, 
                                   basePosition=basePosition, 
                                   baseOrientation=baseOrientation)
        item_ids.append(bodyId)
        
        # Check for immediate overlap
        # Get Contact Points
        # performCollisionDetection happens automatically?
        p.performCollisionDetection()
        contacts = p.getContactPoints(bodyA=bodyId)
        
        # Filter contacts to finding intersections (penetration depth < 0 or > 0 depending on engine?)
        # PyBullet contact distance < 0 means penetration.
        
        for contact in contacts:
            dist = contact[8] # contact distance
            if dist < -0.0001: # Small epsilon
                 other_id = contact[2]
                 if other_id != planeId: # Ignore floor contact
                     print(f"[FAIL] Collision detected! Item {i} intersects Body {other_id} (Dist: {dist})")
                     collision_count += 1
    
    if collision_count == 0:
        print("[PASS] Physics check passed. No items are overlapping.")
    else:
        print(f"[FAIL] {collision_count} collisions detected.")
        
    p.disconnect()

if __name__ == "__main__":
    run_physics_verification()
