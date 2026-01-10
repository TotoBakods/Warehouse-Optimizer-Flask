# IMPORTS
import pybullet as p
import pybullet_data

# ... existing code ...

def physics_settle(solution, items_props, wh_dims, layer_heights=None):
    """
    Refine solution using PyBullet physics engine to ensure physically valid placement.
    Simulates gravity to let items resolve small overlaps and settle.
    """
    try:
        # Initialize Direct (Headless) Simulation
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 1. Environment Setup
        p.loadURDF("plane.urdf")
        
        wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
        
        # Add Walls to keep items inside warehouse
        # (x, y, z, dx, dy, dz) - PyBullet uses half-extents
        wall_thick = 1.0
        
        # Left/Right Walls (X-axis)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2]), basePosition=[-wall_thick, wh_wid/2, wh_hgt/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2]), basePosition=[wh_len+wall_thick, wh_wid/2, wh_hgt/2])
        
        # Front/Back Walls (Y-axis)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2]), basePosition=[wh_len/2, -wall_thick, wh_hgt/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2]), basePosition=[wh_len/2, wh_wid+wall_thick, wh_hgt/2])

        # Add Layer Dividers (static floors/ceilings)
        if layer_heights:
            divider_thick = 0.05 # 10cm total thickness
            for lz in layer_heights:
                if lz <= 0.1 or lz >= wh_hgt - 0.1: continue # Skip bottom floor and very top
                
                # Create a static plate
                # Note: This prevents items from moving UP through it, AND DOWN through it.
                # Items must be spawned correctly above/below it.
                
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wh_wid/2, divider_thick]),
                                  basePosition=[wh_len/2, wh_wid/2, lz])
        
        # 2. Spawn Items
        num_items = len(solution)
        body_ids = []
        
        for i in range(num_items):
            x, y, z, rot = solution[i]
            l, w, h = items_props[i, 0:3]
            mass = items_props[i, 6] if items_props[i, 6] > 0 else 1.0
            
            # Handle Rotation
            # Optimizer stores Rotation in Z-axis degrees (0, 90, 180...)
            # We treat L/W swap manually in logic, but for Physics we can just rotate the body
            # Original dims L,W,H. Rotate around Z.
            
            # Note: The optimizer logic SWAPS L/W in the bounding box based on rotation.
            # But here we should spawn the box with original L,W,H and apply rotation.
            
            colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2])
            
            # Position: Optimizer uses Center? Verified YES in previous step.
            # PyBullet also uses Center.
            start_pos = [x, y, z + h/2] # Shift Z to center (optimizer usually gives bottom-Z)
            
            start_orn = p.getQuaternionFromEuler([0, 0, math.radians(rot)])
            
            bodyId = p.createMultiBody(baseMass=mass,
                                       baseCollisionShapeIndex=colId,
                                       basePosition=start_pos,
                                       baseOrientation=start_orn)
            
            # Set high friction to prevent sliding
            p.changeDynamics(bodyId, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1)
            body_ids.append(bodyId)
            
        # 3. Simulate
        # Run for 100 steps (~0.5 seconds at 240Hz) to resolve overlap/settle
        for _ in range(200):
            p.stepSimulation()
            
        # 4. Harvest Results
        new_solution = solution.copy()
        
        for i, bodyId in enumerate(body_ids):
            pos, orn = p.getBasePositionAndOrientation(bodyId)
            euler = p.getEulerFromQuaternion(orn)
            
            # Convert back to optimizer format
            # Pos is center. Optimizer expects center X,Y but Bottom Z?
            # Existing optimizer code:
            # solution[idx] = [x, y, z, rot]
            # calculate_z_for_item returns BOTTOM z.
            # So we must convert Center Z back to Bottom Z.
            
            l, w, h = items_props[i, 0:3]
            
            # Rot (degrees Z)
            rot_deg = math.degrees(euler[2]) % 360
            # Snap rotation to nearest 90 for cleaner data? 
            # Physics might tilt items. User usually wants rectified packing.
            # We can snap it, but pos should be exact.
            
            new_solution[i, 0] = pos[0]
            new_solution[i, 1] = pos[1]
            new_solution[i, 2] = pos[2] - h/2 # Convert Center-Z to Bottom-Z
            new_solution[i, 3] = rot_deg
            
        p.disconnect()
        return new_solution
        
    except Exception as e:
        print(f"Physics Settle Failed: {e}")
        try: p.disconnect() 
        except: pass
        return solution # Fallback to original
