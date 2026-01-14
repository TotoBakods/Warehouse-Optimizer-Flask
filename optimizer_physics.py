import pybullet as p
import pybullet_data

def physics_settle(solution, items_props, wh_dims, layer_heights=None):
    """Settle items using PyBullet physics to resolve overlaps."""
    client_id = -1
    try:
        # Initialize headless simulation
        print(f"PyBullet: Connecting to DIRECT mode...")
        client_id = p.connect(p.DIRECT)
        print(f"PyBullet: Connected. Client ID: {client_id}")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=client_id)
        p.setTimeStep(1.0/240.0, physicsClientId=client_id)
        
        # Load ground plane
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        
        wh_len, wh_wid, wh_hgt = wh_dims[0], wh_dims[1], wh_dims[2]
        
        # Add walls (PyBullet uses half-extents)
        wall_thick = 1.0
        
        # X-axis walls
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[-wall_thick, wh_wid/2, wh_hgt/2],
                          physicsClientId=client_id)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thick, wh_wid/2, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len+wall_thick, wh_wid/2, wh_hgt/2],
                          physicsClientId=client_id)
        
        # Y-axis walls
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len/2, -wall_thick, wh_hgt/2],
                          physicsClientId=client_id)
        p.createMultiBody(baseMass=0, 
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wall_thick, wh_hgt/2], physicsClientId=client_id), 
                          basePosition=[wh_len/2, wh_wid+wall_thick, wh_hgt/2],
                          physicsClientId=client_id)

        # Add layer dividers
        divider_thick = 0.05 
        
        # Intermediate layers
        if layer_heights:
            for lz in layer_heights:
                if lz <= 0.1: continue  # Skip floor
                if lz >= wh_hgt - 0.01: continue  # Skip ceiling
                
                center_z = lz - divider_thick
                p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wh_wid/2, divider_thick], physicsClientId=client_id),
                                  basePosition=[wh_len/2, wh_wid/2, center_z],
                                  physicsClientId=client_id)

        # Warehouse ceiling
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wh_len/2, wh_wid/2, 1.0], physicsClientId=client_id),
                          basePosition=[wh_len/2, wh_wid/2, wh_hgt + 1.0],
                          physicsClientId=client_id)
        
        # Spawn items
        num_items = len(solution)
        body_ids = []
        
        for i in range(num_items):
            x, y, z, rot = solution[i]
            l, w, h = items_props[i, 0:3]
            mass = items_props[i, 6] if items_props[i, 6] > 0 else 1.0
            
            # Get rotated dimensions for collision shape
            dx, dy, dz = get_rotated_dims_local(l, w, h, int(rot))
            
            colId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[dx/2, dy/2, dz/2], physicsClientId=client_id)
            
            # Position: x,y are centers, z is bottom - convert to center of mass
            start_pos = [x, y, z + dz/2 + 0.01]
            
            # Identity orientation (rotation baked into dimensions)
            start_orn = p.getQuaternionFromEuler([0, 0, 0])
            
            bodyId = p.createMultiBody(baseMass=mass,
                                       baseCollisionShapeIndex=colId,
                                       basePosition=start_pos,
                                       baseOrientation=start_orn,
                                       physicsClientId=client_id)
            
            # High friction + CCD for stable simulation
            p.changeDynamics(bodyId, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1, ccdSweptSphereRadius=0.05, physicsClientId=client_id)
            body_ids.append(bodyId)
            
        # Run simulation
        print(f"PyBullet: Starting simulation ({len(body_ids)} items)...")
        p.setPhysicsEngineParameter(numSolverIterations=100, numSubSteps=20, physicsClientId=client_id) 
        
        # Keep objects active
        for b in body_ids:
            p.changeDynamics(b, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING, physicsClientId=client_id)

        for step in range(1000): # More steps
            if step % 100 == 0:
                print(f"PyBullet: Step {step}/1000")
            p.stepSimulation(physicsClientId=client_id)
        print("PyBullet: Simulation complete.")
            
        # Extract results
        new_solution = solution.copy()
        
        for i, bodyId in enumerate(body_ids):
            pos, orn = p.getBasePositionAndOrientation(bodyId, physicsClientId=client_id)
            
            dx, dy, dz = get_rotated_dims_local(items_props[i, 0], items_props[i, 1], items_props[i, 2], int(solution[i, 3]))
             
            new_solution[i, 0] = pos[0]
            new_solution[i, 1] = pos[1]
            new_solution[i, 2] = pos[2] - dz/2  # Convert center to bottom Z
            new_solution[i, 3] = solution[i, 3]  # Keep original rotation
            
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
    """Returns (dx, dy, dz) based on rotation code 0-5."""
    code = int(rotation_code) % 6
    if code == 0: return l, w, h
    if code == 1: return w, l, h
    if code == 2: return l, h, w
    if code == 3: return h, l, w
    if code == 4: return w, h, l
    if code == 5: return h, w, l
    return l, w, h
