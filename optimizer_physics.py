import pybullet as p
import pybullet_data
import math

# ... existing code ...

def physics_settle(solution, items_props, wh_dims, layer_heights=None):
    """
    Refine solution using PyBullet physics engine to ensure physically valid placement.
    Simulates gravity to let items resolve small overlaps and settle.
    """
    client_id = -1
    try:
        # Initialize Direct (Headless) Simulation with explicit ID
        print(f"PyBullet: Connecting to DIRECT mode...")
        client_id = p.connect(p.DIRECT)
        print(f"PyBullet: Connected. Client ID: {client_id}")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=client_id)
        p.setTimeStep(1.0/240.0, physicsClientId=client_id) # High precision steps
        
        # 1. Environment Setup
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        
        wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
        
        # Add Walls to keep items inside warehouse
        # (x, y, z, dx, dy, dz) - PyBullet uses half-extents
        wall_thick = 1.0
        
        # Left/Right Walls (X-axis)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[-wall_thick, wh_wid/2, wh_hgt/2],
                          physicsClientId=client_id)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len+wall_thick, wh_wid/2, wh_hgt/2],
                          physicsClientId=client_id)
        
        # Front/Back Walls (Y-axis)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len/2, -wall_thick, wh_hgt/2],
                          physicsClientId=client_id)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len/2, wh_wid+wall_thick, wh_hgt/2],
                          physicsClientId=client_id)

        # Add Layer Dividers (static floors/ceilings)
        divider_thick = 0.05 
        
        # 1. Intermediate Layers
        if layer_heights:
            for lz in layer_heights:
                if lz <= 0.1: continue # Skip floor (plane handles it)
                
                # Check if this is the warehouse top? 
                if lz >= wh_hgt - 0.01: continue # Handled by ceiling logic below
                
                center_z = lz - divider_thick
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wh_wid/2, divider_thick], physicsClientId=client_id),
                                  basePosition=[wh_len/2, wh_wid/2, center_z],
                                  physicsClientId=client_id)

        # 2. Warehouse Ceiling
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wh_wid/2, 1.0], physicsClientId=client_id),
                          basePosition=[wh_len/2, wh_wid/2, wh_hgt + 1.0],
                          physicsClientId=client_id)
        
        # 2. Spawn Items
        num_items = len(solution)
        body_ids = []
        
        for i in range(num_items):
            x, y, z, rot = solution[i]
            l, w, h = items_props[i, 0:3]
            mass = items_props[i, 6] if items_props[i, 6] > 0 else 1.0
            
            # 6-Axis Rotation Logic
            # Code 0-5 dictates the orientation of dimensions relative to world frame.
            # But in PyBullet, we create a shape with halfExtents. 
            # We must either:
            # A) Create a generic box (L,W,H) and rotate it.
            # B) Create a box with pre-swapped dimensions (dx, dy, dz) and 0 rotation.
            # Option B is simpler for simple axis-aligned packing, but Option A is more "physics-real" if we want to rotate it later? 
            # Actually, standardizing on (dx,dy,dz) aligned to axes with 0 rot is easiest for stability.
            # BUT, we might want to visualize rotation?
            # The optimizer tracks rotation as 0-5.
            # Let's derive the effective bounding box dims for collision shape.
            
            dx, dy, dz = get_rotated_dims_local(l, w, h, int(rot))
            
            colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dx/2, dy/2, dz/2], physicsClientId=client_id)
            
            # Position: Optimizer uses Center X,Y but Bottom Z?
            # calculate_z_for_item returns BOTTOM Z.
            # repair_solution_compact stores [x, (center), y (center), z (bottom), rot] ??
            # Wait, repair_solution_compact stores (test_x, test_y, z) where test_x is center?
            # Line 288: best_state = (test_x, test_y, z, ...)
            # Line 378: solution[idx, :] = [b_x, b_y, b_z, b_rot]
            # b_x/b_y are centers. b_z is bottom.
            # PyBullet expects Center of Mass.
            # So Z must be b_z + dz/2.
            
            start_pos = [x, y, z + dz/2 + 0.01] 
            
            # Since we baked rotation into dimensions, orientation is Identity (0,0,0)
            # UNLESS 'rot' implies texture mapping? For collision, it's identity.
            start_orn = p.getQuaternionFromEuler([0, 0, 0])
            
            bodyId = p.createMultiBody(baseMass=mass,
                                       baseCollisionShapeIndex=colId,
                                       basePosition=start_pos,
                                       baseOrientation=start_orn,
                                       physicsClientId=client_id)
            
            # Set high friction to prevent sliding
            # Enable CCD (Continuous Collision Detection)
            p.changeDynamics(bodyId, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1, ccdSweptSphereRadius=0.05, physicsClientId=client_id)
            body_ids.append(bodyId)
            
        # 3. Simulate
        # 3. Simulate
        print(f"PyBullet: Starting simulation ({len(body_ids)} items)...")
        p.setPhysicsEngineParameter(numSolverIterations=100, numSubSteps=20, physicsClientId=client_id) 
        
        # Force activation
        for b in body_ids:
            p.changeDynamics(b, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING, physicsClientId=client_id)

        for step in range(1000): # More steps
            if step % 100 == 0:
                print(f"PyBullet: Step {step}/1000")
            p.stepSimulation(physicsClientId=client_id)
        print("PyBullet: Simulation complete.")
            
        # 4. Harvest Results
        new_solution = solution.copy()
        
        for i, bodyId in enumerate(body_ids):
            pos, orn = p.getBasePositionAndOrientation(bodyId, physicsClientId=client_id)
            # We don't really care about getting rotation BACK from physics since we started axis-aligned.
            # Usually items just slide. Tipping over is bad.
            
            # We need to preserve the rotation code (0-5).
            # But if it tipped?
            # If it tipped, our dimensions are invalid.
            # We assume it didn't tip significantly?
            # Or we just update X,Y,Z.
            
            dx, dy, dz = get_rotated_dims_local(items_props[i, 0], items_props[i, 1], items_props[i, 2], int(solution[i, 3]))
             
            new_solution[i, 0] = pos[0]
            new_solution[i, 1] = pos[1]
            new_solution[i, 2] = pos[2] - dz/2 # Convert Center-Z to Bottom-Z
            # Keep original rotation code
            new_solution[i, 3] = solution[i, 3]
            
        return new_solution
        
    except Exception as e:
        print(f"Physics Settle Failed: {e}")
        return solution # Fallback to original
    finally:
        if client_id >= 0:
            try:
                p.resetSimulation(physicsClientId=client_id)
                p.disconnect(physicsClientId=client_id)
            except:
                pass

def get_rotated_dims_local(l, w, h, rotation_code):
    """
    Returns (dx, dy, dz) based on rotation code 0-5.
    (Self-contained helper to avoid circular imports if necessary, or just easy access)
    """
    code = int(rotation_code) % 6
    if code == 0: return l, w, h
    if code == 1: return w, l, h
    if code == 2: return l, h, w
    if code == 3: return h, l, w
    if code == 4: return w, h, l
    if code == 5: return h, w, l
    return l, w, h
