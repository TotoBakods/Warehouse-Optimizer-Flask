def generate_random_items(warehouse_id, count=50):
    import random
    
    clear_data(warehouse_id) # Clear existing
    
    categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Toys', 'Auto Parts']
    fragility_opts = [True, False]
    stackable_opts = [True, False]
    
    for i in range(count):
        # Biased random for realism
        l = round(random.uniform(0.3, 1.5), 2)
        w = round(random.uniform(0.3, 1.5), 2)
        h = round(random.uniform(0.2, 1.0), 2)
        
        weight = round(random.uniform(2.0, 50.0), 1)
        cat = random.choice(categories)
        fragile = random.choice(fragility_opts) if cat in ['Electronics', 'Toys'] else False
        stackable = not fragile and random.random() > 0.3
        
        item = {
            'name': f"Random Item {i+1}",
            'length': l, 'width': w, 'height': h,
            'weight': weight,
            'category': cat,
            'priority': random.choice([1, 2, 3]),
            'fragility': fragile,
            'stackable': stackable,
            'access_freq': round(random.random(), 3),
            'can_rotate': not fragile, # Simple logic
            'x': 0, 'y': 0, 'z': 0, 'rotation': 0
        }
        add_item(item, warehouse_id)
