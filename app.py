import flask
import pandas as pd
import numpy as np
import json
# Trigger reload again 6
import math
import random
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import csv
import os
import gc

from database import (
    init_db, migrate_db, get_all_items, get_item_by_id, get_warehouse_config,
    get_all_warehouses, get_exclusion_zones, save_solution, add_warehouse,
    delete_warehouse, update_warehouse_config, add_item, update_item,
    delete_item, clear_data, add_exclusion_zone, delete_exclusion_zone,
    load_sample_data, create_default_sample_data, get_metrics_history,
    get_item_stats_by_category, load_generated_data
)
from optimizer import (
    GeneticAlgorithm, ExtremalOptimization, HybridOptimizer, get_valid_z_positions
)
from optimizer_physics import physics_settle

app = Flask(__name__)
CORS(app)

# Global state for optimization
optimization_state = {
    'running': False,
    'algorithm': None,
    'progress': 0,
    'current_fitness': 0,
    'best_fitness': 0,
    'best_solution': None,
    'start_time': None,
    'current_warehouse_id': 1,  # Default warehouse
    'message': 'Idle'

}

def finalize_optimization(solution, algorithm, weights, start_time, warehouse_id=1, time_to_best=0):
    if not optimization_state['running']:
        return

    print("Finalizing optimization...")
    with open('thread_debug.log', 'a') as f:
        f.write("Finalizing optimization...\n")

    end_time = time.time()
    try:
        items = get_all_items(warehouse_id)
        warehouse = get_warehouse_config(warehouse_id)
        
        with open('thread_debug.log', 'a') as f:
            f.write("Loaded items and warehouse config\n")

        with open('thread_debug.log', 'a') as f:
            f.write("Loaded items and warehouse config\n")
            
        # --- PyBullet Physics Refinement ---
        try:
             # Prepare props for physics engine
            num_items = len(items)
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
            layer_heights = warehouse.get('layer_heights', [])
            
            print(f"Running PyBullet Physics Settlement with Layers: {layer_heights}...")
            with open('thread_debug.log', 'a') as f: f.write(f"Running PyBullet Physics Settlement with Layers: {layer_heights}...\n")

            # Update solution with physically settled coordinates
            # Only do this if we have a valid numpy solution array
            if isinstance(solution, np.ndarray) and len(solution) > 0:
                solution = physics_settle(solution, items_props, wh_dims, layer_heights)
                print("PyBullet Settlement Complete.")
        except Exception as e:
            print(f"Physics Integration Error: {e}")
            with open('thread_debug.log', 'a') as f: f.write(f"Physics Integration Error: {e}\n")
        # -----------------------------------

        calculator = GeneticAlgorithm()
        final_fitness, space_util, accessibility, stability, grouping = calculator.fitness_function(
            solution, items, warehouse, weights
        )
        
        with open('thread_debug.log', 'a') as f:
            f.write(f"Calculated fitness: {final_fitness}\n")

        save_solution(solution, algorithm, final_fitness, space_util, accessibility,
                      stability, grouping, end_time - start_time, warehouse_id, time_to_best)

        with open('thread_debug.log', 'a') as f:
             f.write("Saved solution to DB\n")

        optimization_state['best_fitness'] = final_fitness
        optimization_state['best_solution'] = solution
        optimization_state['progress'] = 100
        
    except Exception as e:
        import traceback
        with open('thread_debug.log', 'a') as f:
            f.write(f"Error in finalize_optimization: {e}\n")
            f.write(traceback.format_exc())
        print(f"Error verify: {e}")

    time.sleep(1.1)


# --- Flask API Routes ---

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/script.js')
def serve_script():
    return send_file('script.js')


@app.route('/style.css')
def serve_style():
    return send_file('style.css')


@app.route('/api/warehouses', methods=['GET'])
def get_warehouses_api():
    warehouses = get_all_warehouses()
    return jsonify(warehouses)


@app.route('/api/warehouses', methods=['POST'])
def create_warehouse_api():
    data = request.json
    try:
        warehouse_id = add_warehouse(data)
        return jsonify({'success': True, 'id': warehouse_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/warehouses/<int:warehouse_id>', methods=['DELETE'])
def delete_warehouse_api(warehouse_id):
    try:
        delete_warehouse(warehouse_id)
        return jsonify({'success': True})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/warehouses/switch/<int:warehouse_id>', methods=['POST'])
def switch_warehouse(warehouse_id):
    optimization_state['current_warehouse_id'] = warehouse_id
    return jsonify({'success': True, 'current_warehouse_id': warehouse_id})


@app.route('/api/items', methods=['GET'])
def get_items_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    items = get_all_items(warehouse_id)
    return jsonify(items)


@app.route('/api/items', methods=['POST'])
def add_item_api():
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    try:
        add_item(data, warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/items/<item_id>', methods=['PUT'])
def update_item_api(item_id):
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    try:
        update_item(item_id, data, warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/items/<item_id>', methods=['DELETE'])
def delete_item_api(item_id):
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    try:
        delete_item(item_id, warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Stream processing instead of reading entire file to memory
        # Wrap the binary stream in a TextIOWrapper to read line by line
        text_stream = io.TextIOWrapper(file.stream, encoding='utf-8')
        csv_input = csv.reader(text_stream)

        # Clear items for this warehouse first
        clear_data(warehouse_id)
        gc.collect() # Force cleanup after clearing data

        headers = next(csv_input, None)
        if headers:
            print(f"CSV Headers: {headers}")  # Debug log
        
        items_added = 0
        
        for row in csv_input:
            # Support both 12-column (no positions) and 16-column (with positions) formats
            if len(row) >= 12:
                item_data = {
                    'id': row[0], 
                    'name': row[1], 
                    'length': float(row[2]), 
                    'width': float(row[3]),
                    'height': float(row[4]), 
                    'weight': float(row[5]), 
                    'category': row[6],
                    'priority': int(row[7]), 
                    'fragility': 1 if str(row[8]).lower() in ['1', 'true', 'yes'] else 0,
                    'stackable': 1 if str(row[9]).lower() in ['1', 'true', 'yes'] else 0,
                    'access_freq': int(row[10]),
                    'can_rotate': 1 if str(row[11]).lower() in ['1', 'true', 'yes'] else 0,
                }
                
                # Position columns are optional (default to 0)
                if len(row) >= 16:
                    item_data['x'] = float(row[12])
                    item_data['y'] = float(row[13])
                    item_data['z'] = float(row[14])
                    item_data['rotation'] = int(row[15])
                else:
                    item_data['x'] = 0.0
                    item_data['y'] = 0.0
                    item_data['z'] = 0.0
                    item_data['rotation'] = 0
                
                add_item(item_data, warehouse_id)
                items_added += 1
                
        return jsonify({'success': True, 'message': f'CSV data uploaded successfully. {items_added} items added.'})
    except Exception as e:
        import traceback
        print(f"CSV Upload Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    items = get_all_items(warehouse_id)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(['id', 'name', 'length', 'width', 'height', 'weight', 'category',
                     'priority', 'fragility', 'stackable', 'access_freq', 'can_rotate',
                     'x', 'y', 'z', 'rotation'])

    for item in items:
        writer.writerow([
            item['id'], item.get('name', ''), item['length'], item['width'], item['height'],
            item['weight'], item['category'], item['priority'], item['fragility'],
            int(item['stackable']), item['access_freq'], int(item['can_rotate']),
            item['x'], item['y'], item['z'], item['rotation']
        ])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'warehouse_{warehouse_id}_export.csv'
    )


@app.route('/api/export-manifest', methods=['GET'])
def export_manifest():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    items = get_all_items(warehouse_id)
    warehouse = get_warehouse_config(warehouse_id)

    output = io.StringIO()
    writer = csv.writer(output)

    # Manifest Header
    writer.writerow(['Manifest Report', f'Warehouse: {warehouse["name"]}'])
    writer.writerow(['Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(['Item ID', 'Name', 'Category', 'Position X', 'Position Y', 'Position Z', 'Rotation', 'Dimensions (LxWxH)'])

    # Sort by Z, then X, then Y for logical packing order
    sorted_items = sorted(items, key=lambda i: (i['z'], i['x'], i['y']))

    for item in sorted_items:
        writer.writerow([
            item['id'],
            item.get('name', ''),
            item['category'],
            f"{item['x']:.2f}",
            f"{item['y']:.2f}",
            f"{item['z']:.2f}",
            item['rotation'],
            f"{item['length']}x{item['width']}x{item['height']}"
        ])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'warehouse_{warehouse_id}_manifest.csv'
    )


@app.route('/api/load-generated', methods=['POST'])
def load_generated():
    try:
        warehouse_id = optimization_state['current_warehouse_id']
        success, message = load_generated_data(warehouse_id)
        if success:
            return jsonify({'success': True, 'message': f'Loaded {message} generated items'})
        else:
            return jsonify({'success': False, 'message': f'Failed to load data: {message}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/load-sample-data', methods=['POST'])
def load_sample_data_endpoint():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    try:
        success = load_sample_data(warehouse_id)
        if success:
             return jsonify({'success': True, 'message': 'Sample data loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to load sample data'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-data', methods=['POST'])
def clear_data_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    try:
        clear_data(warehouse_id)
        gc.collect()
        return jsonify({'success': True, 'message': 'Data cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/items/delete_all', methods=['DELETE'])
def delete_all_items_endpoint():
    warehouse_id = request.args.get('warehouse_id', 1)
    try:
        # User requested "delete all items". clear_data resets items and results for the warehouse.
        clear_data(warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


def update_progress(progress, avg_fitness, best_fitness, best_solution, space, access, stability, message=None):
    optimization_state['progress'] = progress
    optimization_state['best_fitness'] = best_fitness
    if message:
        optimization_state['message'] = message
    # Only update best_solution if a new valid solution is provided (optimizer throttles updates)
    if best_solution is not None:
        optimization_state['best_solution'] = best_solution

@app.route('/api/optimize/ga', methods=['POST'])
def optimize_ga():
    global optimization_thread
    if optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization already running'})

    data = request.json
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    items = get_all_items(warehouse_id)
    if not items:
        return jsonify({'success': False, 'error': 'No items to optimize'})

    warehouse = get_warehouse_config(warehouse_id)
    
    optimization_state['running'] = True
    optimization_state['progress'] = 0
    optimization_state['algorithm'] = 'GA'
    optimization_state['start_time'] = time.time()
    optimization_state['current_warehouse_id'] = warehouse_id

    def run_optimization():
        with open('thread_debug.log', 'a') as f:
            f.write("Thread started\n")
        print("Thread started")
        
        pop_size = data.get('population_size', 50)
        generations = data.get('generations', 100)
        
        optimizer = GeneticAlgorithm(population_size=pop_size, generations=generations)
        try:
            best_solution, best_fitness, time_to_best = optimizer.optimize(
                items, warehouse, weights, callback=update_progress, optimization_state=optimization_state
            )
            finalize_optimization(best_solution, 'Genetic Algorithm', weights, optimization_state['start_time'], warehouse_id, time_to_best)
            optimization_state['running'] = False
        except Exception as e:
            import traceback
            with open('optimization_debug.log', 'w') as f:
                f.write(f"Optimization failed: {e}\n")
                f.write(traceback.format_exc())
            print(f"Optimization failed: {e}")
            optimization_state['running'] = False

    print("Starting thread...")
    with open('thread_debug.log', 'a') as f:
            f.write("Starting thread...\n")
    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()
    
    gc.collect()

    return jsonify({'success': True})


@app.route('/api/optimize/eo', methods=['POST'])
def optimize_eo():
    global optimization_thread
    if optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization already running'})

    data = request.json
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    items = get_all_items(warehouse_id)
    if not items:
        return jsonify({'success': False, 'error': 'No items to optimize'})

    warehouse = get_warehouse_config(warehouse_id)

    optimization_state['running'] = True
    optimization_state['progress'] = 0
    optimization_state['algorithm'] = 'EO'
    optimization_state['start_time'] = time.time()
    optimization_state['current_warehouse_id'] = warehouse_id

    def run_optimization():
        iterations = data.get('iterations', 1000)
        
        optimizer = ExtremalOptimization(iterations=iterations)
        try:
            best_solution, best_fitness, time_to_best = optimizer.optimize(
                items, warehouse, weights, callback=update_progress, optimization_state=optimization_state
            )
            finalize_optimization(best_solution, 'Extremal Optimization', weights, optimization_state['start_time'], warehouse_id, time_to_best)
            optimization_state['running'] = False
        except Exception as e:
             print(f"Optimization failed: {e}")
             optimization_state['running'] = False

    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()

    return jsonify({'success': True})


@app.route('/api/optimize/ga-eo', methods=['POST'])
def optimize_hybrid():
    global optimization_thread
    if optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization already running'})

    data = request.json
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    items = get_all_items(warehouse_id)
    if not items:
        return jsonify({'success': False, 'error': 'No items to optimize'})

    warehouse = get_warehouse_config(warehouse_id)

    optimization_state['running'] = True
    optimization_state['progress'] = 0
    optimization_state['algorithm'] = 'Hybrid'
    optimization_state['start_time'] = time.time()
    optimization_state['current_warehouse_id'] = warehouse_id

    def run_optimization():
        optimizer = HybridOptimizer(
            ga_generations=data.get('generations', 100),
            eo_iterations=data.get('iterations', 1000)
        )
        try:
            best_solution, best_fitness, time_to_best = optimizer.optimize(
                items, warehouse, weights, callback=update_progress, optimization_state=optimization_state
            )
            finalize_optimization(best_solution, 'Hybrid GA-EO', weights, optimization_state['start_time'], warehouse_id, time_to_best)
            optimization_state['running'] = False
        except Exception as e:
             print(f"Optimization failed: {e}")
             optimization_state['running'] = False

    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()

    return jsonify({'success': True})


@app.route('/api/optimize/eo-ga', methods=['POST'])
def optimize_hybrid_eo_ga():
    """Hybrid optimizer: EO first for exploration, then GA for refinement."""
    global optimization_thread
    if optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization already running'})

    data = request.json
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    items = get_all_items(warehouse_id)
    if not items:
        return jsonify({'success': False, 'error': 'No items to optimize'})

    warehouse = get_warehouse_config(warehouse_id)

    optimization_state['running'] = True
    optimization_state['progress'] = 0
    optimization_state['algorithm'] = 'Hybrid EO-GA'
    optimization_state['start_time'] = time.time()
    optimization_state['current_warehouse_id'] = warehouse_id

    def run_optimization():
        optimizer = HybridOptimizer(
            ga_generations=data.get('generations', 100),
            eo_iterations=data.get('iterations', 1000)
        )
        try:
            best_solution, best_fitness, time_to_best = optimizer.optimize_eo_ga(
                items, warehouse, weights, callback=update_progress, optimization_state=optimization_state
            )
            finalize_optimization(best_solution, 'Hybrid EO-GA', weights, optimization_state['start_time'], warehouse_id, time_to_best)
            optimization_state['running'] = False
        except Exception as e:
             print(f"Optimization failed: {e}")
             optimization_state['running'] = False

    optimization_thread = threading.Thread(target=run_optimization)
    optimization_thread.start()

    return jsonify({'success': True})


    return jsonify({'success': True})


@app.route('/api/optimize/compare', methods=['POST'])
def optimize_compare():
    algorithms = [
        {'name': 'GA', 'endpoint': '/api/optimize/ga', 'params': {'population_size': 30, 'generations': 50}}, # Reduced for speed in comparison
        {'name': 'EO', 'endpoint': '/api/optimize/eo', 'params': {'iterations': 100}},
    ]
    data = request.json
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])
    results = {}

    def run_single_optimization(algo):
        try:
            items = get_all_items(warehouse_id)
            if not items:
                results[algo['name']] = {'error': 'No items'}
                return

            warehouse = get_warehouse_config(warehouse_id)
            start_time = time.time()
            
            solution = None
            fitness = 0
            time_to_best = 0

            if algo['name'] == 'GA':
                optimizer = GeneticAlgorithm(
                    population_size=algo['params']['population_size'],
                    generations=algo['params']['generations']
                )
                solution, fitness, time_to_best = optimizer.optimize(items, warehouse, weights)
            elif algo['name'] == 'EO':
                optimizer = ExtremalOptimization(iterations=algo['params']['iterations'])
                solution, fitness, time_to_best = optimizer.optimize(items, warehouse, weights)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate detailed metrics for the best solution
            ga_calc = GeneticAlgorithm()
            final_fitness, space_util, accessibility, stability, grouping = ga_calc.fitness_function(
                solution, items, warehouse, weights
            )

            # finalize_optimization(solution, algo['name'], weights, start_time, warehouse_id) # Optional: don't save comparison runs to history to avoid clutter? Or do? Let's save.
            save_solution(solution, algo['name'] + "_COMPARE", final_fitness, space_util, accessibility, stability, grouping, execution_time, warehouse_id, time_to_best)

            results[algo['name']] = {
                'fitness': final_fitness,
                'time': execution_time,
                'time_to_best': time_to_best,
                'space_utilization': space_util,
                'accessibility': accessibility,
                'stability': stability,
                'grouping': grouping
            }

        except Exception as e:
            print(f"Error in {algo['name']}: {e}")
            results[algo['name']] = {'error': str(e)}

    # Run sequentially for fair comparison of CPU (or parallel if we want true wall clock, but Python GIL...)
    # Sequential is better for stable timing comparison on single machine
    for algo in algorithms:
        run_single_optimization(algo)

    return jsonify({'success': True, 'results': results})


@app.route('/api/optimize/status', methods=['GET'])
def get_optimization_status():
    return jsonify(optimization_state)


@app.route('/api/optimize/stop', methods=['POST'])
def stop_optimization():
    optimization_state['running'] = False
    optimization_state['start_time'] = None
    return jsonify({'success': True, 'message': 'Optimization stopped'})


@app.route('/api/warehouse/config', methods=['GET'])
def get_warehouse_config_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    config = get_warehouse_config(warehouse_id)
    return jsonify(config)


@app.route('/api/warehouse/config', methods=['PUT'])
def update_warehouse_config_api():
    data = request.json
    try:
        warehouse_id = data.get('id', 1)
        update_warehouse_config(warehouse_id, data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/warehouse/zones', methods=['GET'])
def get_exclusion_zones_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    zones = get_exclusion_zones(warehouse_id)
    return jsonify(zones)


@app.route('/api/warehouse/zones', methods=['POST'])
def add_exclusion_zone_api():
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    try:
        zone_id = add_exclusion_zone(data, warehouse_id)
        return jsonify({'success': True, 'id': zone_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/warehouse/zones/<zone_id>', methods=['DELETE'])
def delete_exclusion_zone_api(zone_id):
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    try:
        delete_exclusion_zone(zone_id, warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/warehouse/zones/<zone_id>', methods=['PUT'])
def update_exclusion_zone_api(zone_id):
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    try:
        update_exclusion_zone(zone_id, data, warehouse_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/metrics/current', methods=['GET'])
def get_current_metrics():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    items = get_all_items(warehouse_id)
    warehouse = get_warehouse_config(warehouse_id)

    if not items or not warehouse:
        return jsonify({'error': 'No data available'})

    ga = GeneticAlgorithm()
    solution = [{'id': item['id'], 'x': item.get('x', 0), 'y': item.get('y', 0),
                 'z': item.get('z', 0), 'rotation': item.get('rotation', 0)}
                for item in items]

    _, space_util, accessibility, stability, grouping = ga.fitness_function(solution, items, warehouse)
    
    cog_x, cog_y, cog_z = ga.calculate_center_of_gravity(solution, {i['id']: i for i in items})

    return jsonify({
        'space_utilization': space_util,
        'accessibility': accessibility,
        'stability': stability,
        'grouping': grouping,
        'total_items': len(items),
        'warehouse_volume': warehouse['length'] * warehouse['width'] * warehouse['height'],
        'center_of_gravity': {'x': cog_x, 'y': cog_y, 'z': cog_z}
    })


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    history = get_metrics_history(warehouse_id)
    return jsonify(history)


@app.route('/api/metrics/categories', methods=['GET'])
def get_category_metrics_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    stats = get_item_stats_by_category(warehouse_id)
    return jsonify(stats)


@app.route('/api/metrics/algo-best', methods=['GET'])
def get_algo_best_performance():
    """Get the best performance for each algorithm type."""
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    
    import sqlite3
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    
    results = []
    
    # Define algorithms to track with their search patterns
    algo_configs = [
        {'name': 'Genetic Algorithm', 'patterns': ['%Genetic Algorithm%', '%GA%'], 'exclude': ['Hybrid', 'EO']},
        {'name': 'Extremal Optimization', 'patterns': ['%Extremal%', '%EO%'], 'exclude': ['Hybrid', 'GA']},
        {'name': 'Hybrid GA+EO', 'patterns': ['%Hybrid GA-EO%', '%GA-EO%', '%ga-eo%']},
        {'name': 'Hybrid EO+GA', 'patterns': ['%Hybrid EO-GA%', '%EO-GA%', '%eo-ga%']},
    ]
    
    for config in algo_configs:
        # Build query for this algorithm
        conditions = []
        params = [warehouse_id]
        
        for pattern in config['patterns']:
            conditions.append('algorithm LIKE ?')
            params.append(pattern)
        
        # Add exclusions for pure algorithms
        exclude_conditions = []
        if 'exclude' in config:
            for exc in config['exclude']:
                exclude_conditions.append(f"algorithm NOT LIKE '%{exc}%'")
        
        where_clause = f"warehouse_id = ? AND ({' OR '.join(conditions)})"
        if exclude_conditions:
            where_clause += f" AND {' AND '.join(exclude_conditions)}"
        
        c.execute(f'''
            SELECT algorithm, fitness, time_to_best, timestamp, execution_time
            FROM optimization_results
            WHERE {where_clause}
            ORDER BY fitness DESC
            LIMIT 1
        ''', params)
        
        row = c.fetchone()
        if row:
            results.append({
                'algorithm': config['name'],
                'best_fitness': row[1],
                'time_to_best': row[2] if row[2] else 0,
                'timestamp': row[3],
                'execution_time': row[4] if row[4] else 0
            })
    
    conn.close()
    return jsonify(results)


@app.route('/api/metrics/algo-best/clear', methods=['POST'])
def clear_algo_best_performance():
    """Clear all optimization results for a warehouse."""
    data = request.json or {}
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])
    
    import sqlite3
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    
    try:
        c.execute('DELETE FROM optimization_results WHERE warehouse_id = ?', (warehouse_id,))
        conn.commit()
        deleted_count = c.rowcount
        conn.close()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'error': str(e)})


# Benchmark state (separate from regular optimization)
benchmark_state = {
    'running': False,
    'progress': 0,
    'current_algo': '',
    'current_run': 0,
    'total_runs': 0,
    'results': {}
}


@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run all algorithms multiple times and calculate averages."""
    global benchmark_state
    
    if benchmark_state['running'] or optimization_state['running']:
        return jsonify({'success': False, 'error': 'Optimization or benchmark already running'})
    
    data = request.json or {}
    runs_per_algo = data.get('runs', 20)
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])
    
    # Reduced parameters for benchmarking (faster)
    benchmark_generations = data.get('generations', 50)
    benchmark_iterations = data.get('iterations', 500)
    benchmark_population = data.get('population_size', 30)
    
    items = get_all_items(warehouse_id)
    if not items:
        return jsonify({'success': False, 'error': 'No items to optimize'})
    
    warehouse = get_warehouse_config(warehouse_id)
    weights = data.get('weights', {'space': 0.5, 'accessibility': 0.4, 'stability': 0.1})
    
    benchmark_state['running'] = True
    benchmark_state['progress'] = 0
    benchmark_state['results'] = {}
    
    algorithms = [
        ('GA', 'Genetic Algorithm', lambda: GeneticAlgorithm(
            population_size=benchmark_population,
            generations=benchmark_generations
        )),
        ('EO', 'Extremal Optimization', lambda: ExtremalOptimization(
            iterations=benchmark_iterations
        )),
        ('GA-EO', 'Hybrid GA-EO', lambda: HybridOptimizer(
            ga_generations=benchmark_generations,
            eo_iterations=benchmark_iterations
        )),
        ('EO-GA', 'Hybrid EO-GA', 'eo-ga'),  # Special case - uses dedicated endpoint logic
    ]
    
    total_runs = len(algorithms) * runs_per_algo
    benchmark_state['total_runs'] = total_runs
    
    def run_benchmark_thread():
        run_count = 0
        
        for algo_key, algo_name, algo_factory in algorithms:
            benchmark_state['current_algo'] = algo_name
            fitness_scores = []
            time_to_best_scores = []
            exec_times = []
            
            for run in range(runs_per_algo):
                if not benchmark_state['running']:
                    return
                
                benchmark_state['current_run'] = run + 1
                run_count += 1
                benchmark_state['progress'] = (run_count / total_runs) * 100
                
                start_time = time.time()
                
                try:
                    # Special handling for EO-GA (EO first, then GA)
                    if algo_key == 'EO-GA':
                        # Phase 1: Run EO
                        eo = ExtremalOptimization(iterations=benchmark_iterations)
                        eo_solution, eo_fitness, eo_ttb = eo.optimize(
                            items, warehouse, weights, callback=None, optimization_state={'running': True}
                        )
                        
                        # Phase 2: Run GA using EO result
                        ga = GeneticAlgorithm(
                            population_size=benchmark_population,
                            generations=benchmark_generations
                        )
                        solution, fitness, ga_ttb = ga.optimize(
                            items, warehouse, weights, callback=None, optimization_state={'running': True},
                            initial_solution=eo_solution
                        )
                        ttb = eo_ttb + ga_ttb
                    else:
                        optimizer = algo_factory()
                        solution, fitness, ttb = optimizer.optimize(
                            items, warehouse, weights, callback=None, optimization_state={'running': True}
                        )
                    
                    exec_time = time.time() - start_time
                    
                    fitness_scores.append(fitness)
                    time_to_best_scores.append(ttb)
                    exec_times.append(exec_time)
                except Exception as e:
                    print(f"Benchmark run failed for {algo_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Calculate averages
            if fitness_scores:
                benchmark_state['results'][algo_key] = {
                    'algorithm': algo_name,
                    'avg_fitness': sum(fitness_scores) / len(fitness_scores),
                    'avg_time_to_best': sum(time_to_best_scores) / len(time_to_best_scores),
                    'avg_execution_time': sum(exec_times) / len(exec_times),
                    'runs': len(fitness_scores),
                    'min_fitness': min(fitness_scores),
                    'max_fitness': max(fitness_scores)
                }
                
                # Save to database as a benchmark result
                save_solution(
                    None, 
                    f"{algo_name} (Benchmark Avg)", 
                    benchmark_state['results'][algo_key]['avg_fitness'],
                    0, 0, 0, 0,
                    benchmark_state['results'][algo_key]['avg_execution_time'],
                    warehouse_id,
                    benchmark_state['results'][algo_key]['avg_time_to_best']
                )
        
        benchmark_state['running'] = False
        benchmark_state['progress'] = 100
    
    import threading
    thread = threading.Thread(target=run_benchmark_thread)
    thread.start()
    
    return jsonify({'success': True, 'total_runs': total_runs})


@app.route('/api/benchmark/status', methods=['GET'])
def get_benchmark_status():
    """Get current benchmark progress."""
    return jsonify({
        'running': benchmark_state['running'],
        'progress': benchmark_state['progress'],
        'current_algo': benchmark_state['current_algo'],
        'current_run': benchmark_state['current_run'],
        'total_runs': benchmark_state['total_runs'],
        'results': benchmark_state['results']
    })


@app.route('/api/benchmark/stop', methods=['POST'])
def stop_benchmark():
    """Stop the benchmark."""
    benchmark_state['running'] = False
    return jsonify({'success': True})


if __name__ == '__main__':
    init_db()
    migrate_db()
    if not get_all_items():
        print("No items found in the database. Loading sample data...")
        load_sample_data()

    app.run(debug=True, host='0.0.0.0', port=5000)