import flask
import pandas as pd
import numpy as np
import json
import sqlite3
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

app = Flask(__name__)
CORS(app)

# Global variables for optimization state
optimization_state = {
    'running': False,
    'algorithm': None,
    'progress': 0,
    'current_fitness': 0,
    'best_fitness': 0,
    'best_solution': None,
    'start_time': None,
    'current_warehouse_id': 1  # Default warehouse
}


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


# Database setup
def init_db():
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS items
                 (id TEXT PRIMARY KEY, name TEXT, length REAL, width REAL, 
                  height REAL, weight REAL, category TEXT, priority INTEGER,
                  fragility INTEGER, stackable INTEGER, access_freq INTEGER,
                  can_rotate INTEGER, x REAL, y REAL, z REAL, rotation INTEGER,
                  warehouse_id INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS warehouse_config
                 (id INTEGER PRIMARY KEY, name TEXT, length REAL, width REAL, height REAL,
                  grid_size REAL, levels INTEGER, walkway_width REAL, layer_heights_json TEXT,
                  is_active INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS exclusion_zones
                 (id INTEGER PRIMARY KEY, name TEXT, x1 REAL, y1 REAL, 
                  x2 REAL, y2 REAL, zone_type TEXT, warehouse_id INTEGER DEFAULT 1)''')

    c.execute('''CREATE TABLE IF NOT EXISTS optimization_results
                 (id INTEGER PRIMARY KEY, algorithm TEXT, fitness REAL,
                  space_utilization REAL, accessibility REAL, stability REAL,
                  grouping REAL, execution_time REAL, timestamp DATETIME, 
                  solution_data TEXT, warehouse_id INTEGER DEFAULT 1)''')

    # Insert default warehouse if it doesn't exist
    c.execute('''INSERT OR IGNORE INTO warehouse_config 
                 (id, name, length, width, height, grid_size, levels, walkway_width, layer_heights_json)
                 VALUES (1, 'Default Warehouse', 10.0, 8.0, 5.0, 0.5, 1, 1.0, '[]')''')

    conn.commit()
    conn.close()


# Load sample data from CSV
def load_sample_data():
    try:
        if os.path.exists('datasets.csv'):
            df = pd.read_csv('datasets.csv')

            column_mapping = {
                'id': 'id',
                'length': 'length',
                'width': 'width',
                'height': 'height',
                'weight': 'weight',
                'category': 'category',
                'priority': 'priority',
                'fragility': 'fragility',
                'stackable': 'stackable',
                'access_freq': 'access_freq',
                'can_rotate': 'can_rotate',
                'x': 'x',
                'y': 'y',
                'z': 'z',
                'rotation': 'rotation'
            }

            db_df = df[list(column_mapping.keys())].rename(columns=column_mapping)

            db_df = db_df.fillna({
                'x': 0, 'y': 0, 'z': 0, 'rotation': 0,
                'name': '', 'category': 'General'
            })

            conn = sqlite3.connect('warehouse.db')
            db_df.to_sql('items', conn, if_exists='replace', index=False)
            conn.close()
            return True
        else:
            return create_default_sample_data()
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return create_default_sample_data()


def create_default_sample_data():
    try:
        sample_data = [
            ['BOX001', 'Small Box', 1.0, 0.8, 0.6, 12.0, 'General', 2, 0, 1, 3, 1, 0, 0, 0, 0],
            ['BOX002', 'Medium Box', 1.2, 1.0, 0.8, 18.5, 'General', 2, 0, 1, 2, 1, 0, 0, 0, 0],
            ['BOX003', 'Large Box', 1.5, 1.2, 1.0, 25.0, 'General', 3, 0, 1, 1, 1, 0, 0, 0, 0],
            ['ELEC001', 'Electronics', 0.8, 0.6, 0.4, 8.2, 'Electronics', 1, 1, 0, 5, 1, 0, 0, 0, 0],
            ['HEAVY001', 'Heavy Item', 2.0, 1.5, 1.0, 45.0, 'Heavy', 3, 0, 1, 1, 0, 0, 0, 0, 0]
        ]

        conn = sqlite3.connect('warehouse.db')
        c = conn.cursor()

        c.execute('DELETE FROM items')

        for item in sample_data:
            c.execute('''INSERT INTO items 
                         (id, name, length, width, height, weight, category, priority,
                          fragility, stackable, access_freq, can_rotate, x, y, z, rotation)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', item)

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating default sample data: {e}")
        return False


# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def __init__(self, population_size=100, generations=500, crossover_rate=0.8,
                 mutation_rate=0.1, selection_method='tournament'):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.initial_population_from_solution = None

    def initialize_population(self, items, warehouse):
        population = []
        if self.initial_population_from_solution:
            population.append(self.initial_population_from_solution)

        while len(population) < self.population_size:
            solution = self.create_random_solution(items, warehouse)
            population.append(solution)
        return population

    def _get_rotated_bounding_box_dims(self, item, rotation):
        rad = math.radians(rotation)
        abs_cos = abs(math.cos(rad))
        abs_sin = abs(math.sin(rad))
        width = item['length'] * abs_cos + item['width'] * abs_sin
        height = item['length'] * abs_sin + item['width'] * abs_cos
        return width, height

    def create_random_solution(self, items, warehouse):
        items_dict = {item['id']: item for item in items}
        solution = []
        placed_items = set()
        valid_z_positions = get_valid_z_positions(warehouse)

        for item in items:
            if item['id'] in placed_items:
                continue

            placed = False
            attempts = 0
            while not placed and attempts < 100:
                rotation = 0
                if item['can_rotate'] and random.random() > 0.5:
                    rotation = random.choice([0, 90, 180, 270])

                bound_w, bound_h = self._get_rotated_bounding_box_dims(item, rotation)

                min_center_x = bound_w / 2
                max_center_x = warehouse['length'] - bound_w / 2
                min_center_y = bound_h / 2
                max_center_y = warehouse['width'] - bound_h / 2

                x = random.uniform(min_center_x, max_center_x) if max_center_x > min_center_x else min_center_x
                y = random.uniform(min_center_y, max_center_y) if max_center_y > min_center_y else min_center_y
                z = 0

                if random.random() > 0.3 and len(solution) > 0:
                    potential_base_item_sol = random.choice(solution)
                    base_item = items_dict.get(potential_base_item_sol['id'])
                    if base_item and base_item['stackable']:
                        z = potential_base_item_sol['z'] + base_item['height']
                else:
                    z = random.choice(valid_z_positions)

                if self.is_valid_placement(item, x, y, z, rotation, solution, warehouse, items_dict):
                    solution.append({
                        'id': item['id'],
                        'x': x, 'y': y, 'z': z,
                        'rotation': rotation
                    })
                    placed_items.add(item['id'])
                    placed = True
                attempts += 1

            if not placed:
                solution.append({
                    'id': item['id'],
                    'x': 0, 'y': 0, 'z': 0,
                    'rotation': 0
                })
                placed_items.add(item['id'])

        return solution

    def is_valid_placement(self, item, x, y, z, rotation, current_solution, warehouse, items_dict):
        if not (z >= 0 and z + item['height'] <= warehouse['height'] + 1e-6):
            return False

        item_vertices = self._get_rotated_vertices(item, x, y, rotation)

        for vx, vy in item_vertices:
            if not (0 <= vx <= warehouse['length'] and 0 <= vy <= warehouse['width']):
                return False

        exclusion_zones = get_exclusion_zones(warehouse['id'])
        for zone in exclusion_zones:
            zone_vertices = [
                (zone['x1'], zone['y1']), (zone['x2'], zone['y1']),
                (zone['x2'], zone['y2']), (zone['x1'], zone['y2'])
            ]
            if self._check_polygon_collision(item_vertices, zone_vertices):
                return False

        for placed_item_sol in current_solution:
            placed_item = items_dict.get(placed_item_sol['id'])
            if placed_item and placed_item['id'] != item['id']:
                if self.check_collision(item, x, y, z, rotation,
                                        placed_item, placed_item_sol['x'], placed_item_sol['y'],
                                        placed_item_sol['z'], placed_item_sol['rotation']):
                    return False
        return True

    def _get_rotated_vertices(self, item, center_x, center_y, rotation):
        length = item['length']
        width = item['width']

        unrotated_corners = [
            (-length / 2, -width / 2),
            (length / 2, -width / 2),
            (length / 2, width / 2),
            (-length / 2, width / 2)
        ]

        rad = math.radians(rotation)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)

        rotated_corners = []
        for corner_x, corner_y in unrotated_corners:
            rotated_x = corner_x * cos_rad - corner_y * sin_rad
            rotated_y = corner_x * sin_rad + corner_y * cos_rad

            final_x = center_x + rotated_x
            final_y = center_y + rotated_y
            rotated_corners.append((final_x, final_y))

        return rotated_corners

    def get_rotated_dimensions(self, item, rotation):
        if rotation in [90, 270]:
            return item['width'], item['length']
        return item['length'], item['width']

    def _get_axes(self, vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            axes.append(normal)
        return [axes[0], axes[1]]

    def _project_vertices(self, vertices, axis):
        axis_mag = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
        if axis_mag == 0: return 0, 0
        axis_unit = (axis[0] / axis_mag, axis[1] / axis_mag)

        projections = [v[0] * axis_unit[0] + v[1] * axis_unit[1] for v in vertices]
        return min(projections), max(projections)

    def _check_polygon_collision(self, vertices1, vertices2):
        axes = self._get_axes(vertices1) + self._get_axes(vertices2)

        for axis in axes:
            min1, max1 = self._project_vertices(vertices1, axis)
            min2, max2 = self._project_vertices(vertices2, axis)

            if max1 < min2 or max2 < min1:
                return False

        return True

    def check_collision(self, item1, x1, y1, z1, rotation1, item2, x2, y2, z2, rotation2):
        if not (z1 < z2 + item2['height'] and z1 + item1['height'] > z2):
            return False

        vertices1 = self._get_rotated_vertices(item1, x1, y1, rotation1)
        vertices2 = self._get_rotated_vertices(item2, x2, y2, rotation2)

        return self._check_polygon_collision(vertices1, vertices2)

    def fitness_function(self, solution, items, warehouse, weights=None):
        if weights is None:
            weights = {'space': 0.6, 'accessibility': 0.3, 'stability': 0.1, 'grouping': 0.0}

        space_utilization = self.calculate_space_utilization(solution, items, warehouse)
        accessibility = self.calculate_accessibility(solution, items, warehouse)
        stability = self.calculate_stability(solution, items, warehouse)
        grouping = self.calculate_grouping(solution, items, warehouse)

        total_weight = sum(weights.values())
        if total_weight == 0: total_weight = 1

        norm_weights = {k: v / total_weight for k, v in weights.items()}

        fitness = (norm_weights.get('space', 0) * space_utilization +
                   norm_weights.get('accessibility', 0) * accessibility +
                   norm_weights.get('stability', 0) * stability +
                   norm_weights.get('grouping', 0) * grouping)

        return fitness, space_utilization, accessibility, stability, grouping

    def calculate_space_utilization(self, solution, items, warehouse):
        total_item_volume = 0
        for item in items:
            total_item_volume += item['length'] * item['width'] * item['height']

        warehouse_volume = warehouse['length'] * warehouse['width'] * warehouse['height']
        return total_item_volume / warehouse_volume if warehouse_volume > 0 else 0

    def calculate_accessibility(self, solution, items, warehouse):
        if not solution:
            return 0

        total_score = 0
        items_dict = {item['id']: item for item in items}
        for item_solution in solution:
            item = items_dict.get(item_solution['id'])
            if item:
                center_x = item_solution['x']
                center_y = item_solution['y']

                dist_x = min(center_x, warehouse['length'] - center_x)
                dist_y = min(center_y, warehouse['width'] - center_y)

                distance_to_edge = min(dist_x, dist_y)

                accessibility_score = 1.0 / (1.0 + distance_to_edge) if distance_to_edge >= 0 else 0
                total_score += accessibility_score * item['access_freq']
        return total_score / len(solution) if solution else 0

    def calculate_stability(self, solution, items, warehouse):
        if not solution:
            return 0

        stable_items = 0
        items_dict = {item['id']: item for item in items}
        valid_z_positions = get_valid_z_positions(warehouse)
        z_tolerance = 1e-6

        for item_solution in solution:
            item_above = items_dict.get(item_solution['id'])
            if not item_above: continue

            is_on_layer = any(abs(item_solution['z'] - z_pos) < z_tolerance for z_pos in valid_z_positions)

            if is_on_layer:
                stable_items += 1
                continue

            is_supported = False
            for item_below_solution in solution:
                if item_below_solution['id'] != item_solution['id']:
                    item_below = items_dict.get(item_below_solution['id'])

                    if item_below and item_below['stackable'] and \
                            abs((item_below_solution['z'] + item_below['height']) - item_solution['z']) < z_tolerance:

                        if self.check_collision(
                                item_above, item_solution['x'], item_solution['y'], 0, item_solution['rotation'],
                                item_below, item_below_solution['x'], item_below_solution['y'], 0,
                                item_below_solution['rotation']
                        ):
                            is_supported = True
                            break
            if is_supported:
                stable_items += 1

        return stable_items / len(solution) if solution else 0

    def calculate_grouping(self, solution, items, warehouse):
        if not solution or len(solution) < 2:
            return 1.0

        items_dict = {item['id']: item for item in items}

        items_by_category = {}
        for item_sol in solution:
            item_details = items_dict.get(item_sol['id'])
            if item_details and 'category' in item_details:
                category = item_details['category']
                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(item_sol)

        total_inverse_distance = 0
        pair_count = 0

        for category, cat_items in items_by_category.items():
            if len(cat_items) > 1:
                for i in range(len(cat_items)):
                    for j in range(i + 1, len(cat_items)):
                        item1 = cat_items[i]
                        item2 = cat_items[j]

                        distance = math.sqrt(
                            (item1['x'] - item2['x']) ** 2 +
                            (item1['y'] - item2['y']) ** 2 +
                            (item1['z'] - item2['z']) ** 2
                        )

                        total_inverse_distance += 1.0 / (1.0 + distance)
                        pair_count += 1

        if pair_count == 0:
            return 1.0

        return total_inverse_distance / pair_count

    def selection(self, population, fitness_scores):
        if self.selection_method == 'tournament':
            return self.tournament_selection(population, fitness_scores)
        elif self.selection_method == 'roulette':
            return self.roulette_selection(population, fitness_scores)
        else:
            return self.rank_selection(population, fitness_scores)

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament = [(population[i], fitness_scores[i]) for i in tournament_indices]
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def roulette_selection(self, population, fitness_scores):
        min_fitness = min(fitness_scores) if fitness_scores else 0
        adjusted_fitness = [f - min_fitness + 0.1 for f in fitness_scores]
        total_fitness = sum(adjusted_fitness)

        if total_fitness == 0:
            return random.choices(population, k=len(population))

        probabilities = [f / total_fitness for f in adjusted_fitness]
        return random.choices(population, weights=probabilities, k=len(population))

    def rank_selection(self, population, fitness_scores):
        ranked = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        ranks = list(range(1, len(population) + 1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        return random.choices([p[0] for p in ranked], weights=probabilities, k=len(population))

    def crossover(self, parent1, parent2, items, warehouse):
        if random.random() > self.crossover_rate:
            return parent1, parent2

        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1, parent2

        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return self.repair_solution(child1, items, warehouse), self.repair_solution(child2, items, warehouse)

    def mutation(self, solution, items, warehouse):
        items_dict = {item['id']: item for item in items}
        valid_z_positions = get_valid_z_positions(warehouse)
        for i in range(len(solution)):
            if random.random() < self.mutation_rate:
                item = items_dict.get(solution[i]['id'])
                if item:
                    attempts = 0
                    while attempts < 10:
                        new_rotation = solution[i]['rotation']
                        if item['can_rotate'] and random.random() > 0.5:
                            new_rotation = random.choice([0, 90, 180, 270])

                        bound_w, bound_h = self._get_rotated_bounding_box_dims(item, new_rotation)

                        min_center_x = bound_w / 2
                        max_center_x = warehouse['length'] - bound_w / 2
                        min_center_y = bound_h / 2
                        max_center_y = warehouse['width'] - bound_h / 2

                        new_x = random.uniform(min_center_x,
                                               max_center_x) if max_center_x > min_center_x else min_center_x
                        new_y = random.uniform(min_center_y,
                                               max_center_y) if max_center_y > min_center_y else min_center_y

                        new_z = random.choice(valid_z_positions)

                        temp_solution = [s for j, s in enumerate(solution) if j != i]

                        if self.is_valid_placement(item, new_x, new_y, new_z, new_rotation, temp_solution, warehouse,
                                                   items_dict):
                            solution[i]['x'] = new_x
                            solution[i]['y'] = new_y
                            solution[i]['z'] = new_z
                            solution[i]['rotation'] = new_rotation
                            break
                        attempts += 1

        return solution

    def repair_solution(self, solution, items, warehouse):
        items_dict = {item['id']: item for item in items}

        all_item_ids = {item['id'] for item in items}
        solution_items = {s['id']: s for s in solution}

        initial_placements = []
        for item_id in all_item_ids:
            if item_id in solution_items:
                initial_placements.append(solution_items[item_id])
            else:
                initial_placements.append({'id': item_id, 'x': 0, 'y': 0, 'z': 0, 'rotation': 0})

        final_solution = []
        for item_sol_to_check in initial_placements:
            item = items_dict.get(item_sol_to_check['id'])
            if not item:
                continue

            if self.is_valid_placement(item, item_sol_to_check['x'], item_sol_to_check['y'], item_sol_to_check['z'],
                                       item_sol_to_check['rotation'], final_solution, warehouse, items_dict):
                final_solution.append(item_sol_to_check)
            else:
                placed = False
                for _ in range(100):
                    rotation = item_sol_to_check['rotation']
                    if item['can_rotate']:
                        rotation = random.choice([0, 90, 180, 270])

                    bound_w, bound_h = self._get_rotated_bounding_box_dims(item, rotation)

                    min_center_x = bound_w / 2
                    max_center_x = warehouse['length'] - bound_w / 2
                    min_center_y = bound_h / 2
                    max_center_y = warehouse['width'] - bound_h / 2

                    x = random.uniform(min_center_x, max_center_x) if max_center_x > min_center_x else min_center_x
                    y = random.uniform(min_center_y, max_center_y) if max_center_y > min_center_y else min_center_y
                    z = 0

                    if self.is_valid_placement(item, x, y, z, rotation, final_solution, warehouse, items_dict):
                        final_solution.append({'id': item['id'], 'x': x, 'y': y, 'z': z, 'rotation': rotation})
                        placed = True
                        break

                if not placed:
                    final_solution.append({'id': item['id'], 'x': 0, 'y': 0, 'z': 0, 'rotation': 0})

        return final_solution

    def optimize(self, items, warehouse, weights=None, callback=None):
        population = self.initialize_population(items, warehouse)
        best_solution = None
        best_fitness = -float('inf')

        for generation in range(self.generations):
            if not optimization_state['running']:
                break

            fitness_scores = []
            space_utils = []
            access_scores = []
            stability_scores = []

            for solution in population:
                fitness, space_util, accessibility, stability, _ = self.fitness_function(solution, items, warehouse,
                                                                                         weights)
                fitness_scores.append(fitness)
                space_utils.append(space_util)
                access_scores.append(accessibility)
                stability_scores.append(stability)

            current_best_idx = np.argmax(fitness_scores) if fitness_scores else 0
            if fitness_scores and fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_solution = population[current_best_idx].copy()

            selected = self.selection(population, fitness_scores)

            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i + 1], items, warehouse)
                    new_population.extend([self.mutation(child1, items, warehouse),
                                           self.mutation(child2, items, warehouse)])
                else:
                    new_population.append(self.mutation(selected[i], items, warehouse))

            population = new_population

            if callback and generation % 10 == 0:
                progress = (generation + 1) / self.generations * 100
                avg_fitness = np.mean(fitness_scores) if fitness_scores else 0
                callback(progress, avg_fitness, best_fitness, best_solution,
                         np.mean(space_utils) if space_utils else 0,
                         np.mean(access_scores) if access_scores else 0,
                         np.mean(stability_scores) if stability_scores else 0)

        if callback and optimization_state['running']:
            final_fitness, final_space, final_access, final_stability, _ = self.fitness_function(best_solution, items,
                                                                                                 warehouse, weights)
            callback(100, final_fitness, final_fitness, best_solution,
                     final_space, final_access, final_stability)

        return best_solution, best_fitness


# Extremal Optimization Implementation
class ExtremalOptimization:
    def __init__(self, iterations=1000, tau=1.5):
        self.iterations = iterations
        self.tau = tau
        self.initial_solution = None

    def optimize(self, items, warehouse, weights=None, callback=None):
        items_dict = {item['id']: item for item in items}
        ga = GeneticAlgorithm()

        if self.initial_solution:
            current_solution = self.initial_solution
        else:
            current_solution = ga.create_random_solution(items, warehouse)

        best_solution = current_solution.copy()
        best_fitness, _, _, _, _ = ga.fitness_function(best_solution, items, warehouse, weights)
        current_fitness = best_fitness

        for iteration in range(self.iterations):
            if not optimization_state['running']:
                break

            item_fitnesses = []
            for i, item_sol in enumerate(current_solution):
                temp_solution = [s for j, s in enumerate(current_solution) if j != i]
                item = items_dict.get(item_sol['id'])
                if item:
                    fitness_contribution = self.calculate_item_fitness(
                        item, item_sol, temp_solution, items, warehouse, weights)
                    item_fitnesses.append((i, fitness_contribution))

            if not item_fitnesses:
                break

            item_fitnesses.sort(key=lambda x: x[1])

            weights_eo = [(rank + 1) ** -self.tau for rank in range(len(item_fitnesses))]
            chosen_item_fitness = random.choices(item_fitnesses, weights=weights_eo, k=1)[0]
            selected_idx = chosen_item_fitness[0]

            current_solution = self.perturb_item(
                current_solution, selected_idx, items, warehouse, items_dict)

            current_fitness, current_space, current_access, current_stability, _ = ga.fitness_function(
                current_solution, items, warehouse, weights)

            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

            progress = (iteration + 1) / self.iterations * 100
            if callback and (iteration % 10 == 0 or iteration == self.iterations - 1):
                callback(progress, current_fitness, best_fitness, best_solution,
                         current_space, current_access, current_stability)

        if callback and optimization_state['running']:
            final_fitness, final_space, final_access, final_stability, _ = ga.fitness_function(best_solution, items,
                                                                                               warehouse, weights)
            callback(100, final_fitness, final_fitness, best_solution,
                     final_space, final_access, final_stability)

        return best_solution, best_fitness

    def calculate_item_fitness(self, item, item_solution, other_solution, items, warehouse, weights=None):
        if weights is None:
            weights = {'space': 0.6, 'accessibility': 0.3, 'stability': 0.1, 'grouping': 0.1}

        items_dict = {i['id']: i for i in items}

        space_contrib = item['length'] * item['width'] * item['height']

        center_x = item_solution['x']
        center_y = item_solution['y']
        dist_x = min(center_x, warehouse['length'] - center_x)
        dist_y = min(center_y, warehouse['width'] - center_y)
        distance_to_edge = min(dist_x, dist_y)
        access_contrib = 1.0 / (1.0 + distance_to_edge) if distance_to_edge >= 0 else 0

        stability_contrib = 1.0 if item_solution['z'] == 0 else 0.5

        grouping_contrib = 0
        same_category_count = 0
        item_category = items_dict.get(item['id'], {}).get('category')
        if item_category:
            for other_item_sol in other_solution:
                other_item = items_dict.get(other_item_sol['id'])
                if other_item and other_item.get('category') == item_category:
                    distance = math.sqrt(
                        (item_solution['x'] - other_item_sol['x']) ** 2 +
                        (item_solution['y'] - other_item_sol['y']) ** 2 +
                        (item_solution['z'] - other_item_sol['z']) ** 2
                    )
                    grouping_contrib += 1.0 / (1.0 + distance)
                    same_category_count += 1

        if same_category_count > 0:
            grouping_contrib /= same_category_count
        else:
            grouping_contrib = 1.0

        total_weight = sum(weights.values())
        if total_weight == 0: total_weight = 1
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        return (norm_weights.get('space', 0) * space_contrib +
                norm_weights.get('accessibility', 0) * access_contrib +
                norm_weights.get('stability', 0) * stability_contrib +
                norm_weights.get('grouping', 0) * grouping_contrib)

    def perturb_item(self, solution, item_idx, items, warehouse, items_dict):
        ga = GeneticAlgorithm()
        valid_z_positions = get_valid_z_positions(warehouse)
        item_sol = solution[item_idx]
        item = items_dict.get(item_sol['id'])

        if not item:
            return solution

        attempts = 0
        while attempts < 20:
            new_rotation = item_sol['rotation']
            if item['can_rotate']:
                new_rotation = random.choice([0, 90, 180, 270])

            bound_w, bound_h = ga._get_rotated_bounding_box_dims(item, new_rotation)

            min_center_x = bound_w / 2
            max_center_x = warehouse['length'] - bound_w / 2
            min_center_y = bound_h / 2
            max_center_y = warehouse['width'] - bound_h / 2

            new_x = random.uniform(min_center_x, max_center_x) if max_center_x > min_center_x else min_center_x
            new_y = random.uniform(min_center_y, max_center_y) if max_center_y > min_center_y else min_center_y
            new_z = random.choice(valid_z_positions)

            temp_solution = [s for j, s in enumerate(solution) if j != item_idx]

            if ga.is_valid_placement(item, new_x, new_y, new_z, new_rotation, temp_solution, warehouse, items_dict):
                solution[item_idx]['x'] = new_x
                solution[item_idx]['y'] = new_y
                solution[item_idx]['z'] = new_z
                solution[item_idx]['rotation'] = new_rotation
                break

            attempts += 1

        return solution


# Hybrid Algorithms
class HybridOptimizer:
    @staticmethod
    def ga_then_eo(items, warehouse, ga_generations=250, eo_iterations=500, weights=None, callback=None):
        ga = GeneticAlgorithm(population_size=50, generations=ga_generations)

        def ga_callback(progress, current_fitness, best_fitness, best_sol, space, access, stability):
            if callback:
                callback(progress / 2, current_fitness, best_fitness, best_sol, space, access, stability)

        ga_solution, ga_fitness = ga.optimize(items, warehouse, weights, ga_callback)

        if not optimization_state['running']: return None, 0

        eo = ExtremalOptimization(iterations=eo_iterations)

        def eo_callback(progress, current_fitness, best_fitness, best_sol, space, access, stability):
            if callback:
                callback(50 + progress / 2, current_fitness, best_fitness, best_sol, space, access, stability)

        eo.initial_solution = ga_solution
        final_solution, final_fitness = eo.optimize(items, warehouse, weights, eo_callback)

        return final_solution, final_fitness

    @staticmethod
    def eo_then_ga(items, warehouse, eo_iterations=500, ga_generations=250, weights=None, callback=None):
        eo = ExtremalOptimization(iterations=eo_iterations)

        def eo_callback(progress, current_fitness, best_fitness, best_sol, space, access, stability):
            if callback:
                callback(progress / 2, current_fitness, best_fitness, best_sol, space, access, stability)

        eo_solution, eo_fitness = eo.optimize(items, warehouse, weights, eo_callback)

        if not optimization_state['running']: return None, 0

        ga = GeneticAlgorithm(population_size=50, generations=ga_generations)

        def ga_callback(progress, current_fitness, best_fitness, best_sol, space, access, stability):
            if callback:
                callback(50 + progress / 2, current_fitness, best_fitness, best_sol, space, access, stability)

        ga.initial_population_from_solution = eo_solution
        final_solution, final_fitness = ga.optimize(items, warehouse, weights, ga_callback)

        return final_solution, final_fitness


# Database helper functions
def get_all_items(warehouse_id=None):
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    if warehouse_id:
        c.execute('SELECT * FROM items WHERE warehouse_id = ?', (warehouse_id,))
    else:
        c.execute('SELECT * FROM items')

    rows = c.fetchall()
    conn.close()

    items = []
    for row in rows:
        items.append({
            'id': row[0], 'name': row[1], 'length': row[2], 'width': row[3],
            'height': row[4], 'weight': row[5], 'category': row[6], 'priority': row[7],
            'fragility': row[8], 'stackable': bool(row[9]), 'access_freq': row[10],
            'can_rotate': bool(row[11]), 'x': row[12], 'y': row[13], 'z': row[14],
            'rotation': row[15], 'warehouse_id': row[16] if len(row) > 16 else 1
        })
    return items


def get_item_by_id(item_id):
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute('SELECT * FROM items WHERE id = ?', (item_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {
            'id': row[0], 'name': row[1], 'length': row[2], 'width': row[3],
            'height': row[4], 'weight': row[5], 'category': row[6], 'priority': row[7],
            'fragility': row[8], 'stackable': bool(row[9]), 'access_freq': row[10],
            'can_rotate': bool(row[11]), 'warehouse_id': row[16] if len(row) > 16 else 1
        }
    return None


def get_warehouse_config(warehouse_id=1):
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute('SELECT * FROM warehouse_config WHERE id = ?', (warehouse_id,))
    row = c.fetchone()
    conn.close()

    if row:
        config = {
            'id': row[0],
            'name': row[1],
            'length': row[2], 'width': row[3], 'height': row[4],
            'grid_size': row[5], 'levels': row[6], 'walkway_width': row[7]
        }
        layer_heights_json = row[8] if len(row) > 8 else None
        if layer_heights_json:
            try:
                config['layer_heights'] = json.loads(layer_heights_json)
            except (json.JSONDecodeError, TypeError):
                config['layer_heights'] = []
        else:
            config['layer_heights'] = []
        return config
    return None


def get_all_warehouses():
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute('SELECT * FROM warehouse_config')
    rows = c.fetchall()
    conn.close()

    warehouses = []
    for row in rows:
        warehouse = {
            'id': row[0],
            'name': row[1],
            'length': row[2], 'width': row[3], 'height': row[4],
            'grid_size': row[5], 'levels': row[6], 'walkway_width': row[7],
            'is_active': row[9] if len(row) > 9 else 1
        }
        layer_heights_json = row[8] if len(row) > 8 else None
        if layer_heights_json:
            try:
                warehouse['layer_heights'] = json.loads(layer_heights_json)
            except (json.JSONDecodeError, TypeError):
                warehouse['layer_heights'] = []
        else:
            warehouse['layer_heights'] = []
        warehouses.append(warehouse)
    return warehouses


def get_exclusion_zones(warehouse_id=1):
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute('SELECT * FROM exclusion_zones WHERE warehouse_id = ?', (warehouse_id,))
    rows = c.fetchall()
    conn.close()

    zones = []
    for row in rows:
        zones.append({
            'id': row[0], 'name': row[1], 'x1': row[2], 'y1': row[3],
            'x2': row[4], 'y2': row[5], 'zone_type': row[6],
            'warehouse_id': row[7] if len(row) > 7 else 1
        })
    return zones


def save_solution(solution, algorithm, fitness, space_util, accessibility, stability, grouping, exec_time,
                  warehouse_id=1):
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    if solution:
        for item_sol in solution:
            c.execute('''UPDATE items SET x = ?, y = ?, z = ?, rotation = ? 
                         WHERE id = ? AND warehouse_id = ?''',
                      (item_sol['x'], item_sol['y'], item_sol['z'], item_sol['rotation'],
                       item_sol['id'], warehouse_id))

    c.execute('''INSERT INTO optimization_results 
                 (algorithm, fitness, space_utilization, accessibility, stability, 
                  grouping, execution_time, timestamp, solution_data, warehouse_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (algorithm, fitness, space_util, accessibility, stability, grouping,
               exec_time, datetime.now(), json.dumps(solution), warehouse_id))

    conn.commit()
    conn.close()


def finalize_optimization(solution, algorithm, weights, start_time, warehouse_id=1):
    if not optimization_state['running']:
        return

    end_time = time.time()
    items = get_all_items(warehouse_id)
    warehouse = get_warehouse_config(warehouse_id)

    calculator = GeneticAlgorithm()
    final_fitness, space_util, accessibility, stability, grouping = calculator.fitness_function(
        solution, items, warehouse, weights
    )

    save_solution(solution, algorithm, final_fitness, space_util, accessibility,
                  stability, grouping, end_time - start_time, warehouse_id)

    optimization_state['best_fitness'] = final_fitness
    optimization_state['best_solution'] = solution
    optimization_state['progress'] = 100

    time.sleep(1.1)


# --- Flask API Routes ---

@app.route('/api/warehouses', methods=['GET'])
def get_warehouses():
    warehouses = get_all_warehouses()
    return jsonify(warehouses)


@app.route('/api/warehouses', methods=['POST'])
def create_warehouse():
    data = request.json
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        name = data.get('name', 'New Warehouse')
        length = data.get('length', 10.0)
        width = data.get('width', 8.0)
        height = data.get('height', 5.0)
        grid_size = data.get('grid_size', 0.5)
        levels = data.get('levels', 1)
        walkway_width = data.get('walkway_width', 1.0)
        layer_heights = data.get('layer_heights', [])
        layer_heights_json = json.dumps(layer_heights)

        c.execute('''INSERT INTO warehouse_config 
                     (name, length, width, height, grid_size, levels, walkway_width, layer_heights_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (name, length, width, height, grid_size, levels, walkway_width, layer_heights_json))

        warehouse_id = c.lastrowid
        conn.commit()
        return jsonify({'success': True, 'id': warehouse_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/warehouses/<int:warehouse_id>', methods=['DELETE'])
def delete_warehouse(warehouse_id):
    if warehouse_id == 1:
        return jsonify({'success': False, 'error': 'Cannot delete default warehouse'}), 400

    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        # Delete warehouse config
        c.execute('DELETE FROM warehouse_config WHERE id = ?', (warehouse_id,))
        # Delete associated items
        c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
        # Delete associated exclusion zones
        c.execute('DELETE FROM exclusion_zones WHERE warehouse_id = ?', (warehouse_id,))
        # Delete associated optimization results
        c.execute('DELETE FROM optimization_results WHERE warehouse_id = ?', (warehouse_id,))

        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/warehouses/switch/<int:warehouse_id>', methods=['POST'])
def switch_warehouse(warehouse_id):
    optimization_state['current_warehouse_id'] = warehouse_id
    return jsonify({'success': True, 'current_warehouse_id': warehouse_id})


@app.route('/api/items', methods=['GET'])
def get_items():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    items = get_all_items(warehouse_id)
    return jsonify(items)


@app.route('/api/items', methods=['POST'])
def add_item():
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        c.execute('''INSERT OR REPLACE INTO items 
                     (id, name, length, width, height, weight, category, priority,
                      fragility, stackable, access_freq, can_rotate, x, y, z, rotation, warehouse_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data['id'], data.get('name', ''), data['length'], data['width'],
                   data['height'], data.get('weight', 0), data.get('category', ''),
                   data.get('priority', 1), data.get('fragility', 0),
                   int(data.get('stackable', 1)), data.get('access_freq', 1),
                   int(data.get('can_rotate', 1)), data.get('x', 0), data.get('y', 0),
                   data.get('z', 0), data.get('rotation', 0), warehouse_id))

        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/items/<item_id>', methods=['PUT'])
def update_item(item_id):
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        fields = []
        values = []
        for key, value in data.items():
            if key != 'id' and key != 'warehouse_id':
                fields.append(f"{key} = ?")
                values.append(value)

        values.append(item_id)
        values.append(warehouse_id)
        query = f"UPDATE items SET {', '.join(fields)} WHERE id = ? AND warehouse_id = ?"
        c.execute(query, values)

        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/items/<item_id>', methods=['DELETE'])
def delete_item(item_id):
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        c.execute('DELETE FROM items WHERE id = ? AND warehouse_id = ?', (item_id, warehouse_id))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.reader(stream)

        conn = sqlite3.connect('warehouse.db')
        c = conn.cursor()

        # Only delete items for this specific warehouse
        c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))

        headers = next(csv_input, None)
        for row in csv_input:
            if len(row) >= 16:
                row[8] = 1 if str(row[8]).lower() in ['1', 'true', 'yes'] else 0
                row[9] = 1 if str(row[9]).lower() in ['1', 'true', 'yes'] else 0
                row[11] = 1 if str(row[11]).lower() in ['1', 'true', 'yes'] else 0

                c.execute('''INSERT INTO items 
                             (id, name, length, width, height, weight, category, priority,
                              fragility, stackable, access_freq, can_rotate, x, y, z, rotation, warehouse_id)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          row[:16] + [warehouse_id])

        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'CSV data uploaded successfully'})
    except Exception as e:
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


@app.route('/api/load-sample-data', methods=['POST'])
def load_sample_data_endpoint():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    try:
        success = load_sample_data()
        if success:
            # Update warehouse_id for loaded items
            conn = sqlite3.connect('warehouse.db')
            c = conn.cursor()
            c.execute('UPDATE items SET warehouse_id = ?', (warehouse_id,))
            conn.commit()
            conn.close()

            return jsonify({'success': True, 'message': 'Sample data loaded successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to load sample data'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        c.execute('DELETE FROM items WHERE warehouse_id = ?', (warehouse_id,))
        c.execute('DELETE FROM optimization_results WHERE warehouse_id = ?', (warehouse_id,))
        conn.commit()
        return jsonify({'success': True, 'message': 'Data cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


# Optimization endpoints
@app.route('/api/optimize/ga', methods=['POST'])
def optimize_ga():
    data = request.json
    population_size = data.get('population_size', 50)
    generations = data.get('generations', 100)
    weights = data.get('weights', {})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    def callback(progress, current_fitness, best_fitness, best_solution, space, access, stability):
        optimization_state['progress'] = progress
        optimization_state['current_fitness'] = current_fitness
        optimization_state['best_fitness'] = best_fitness
        optimization_state['best_solution'] = best_solution

    def run_optimization():
        optimization_state['running'] = True
        optimization_state['algorithm'] = 'GA'
        optimization_state['progress'] = 0
        optimization_state['start_time'] = time.time()
        optimization_state['current_warehouse_id'] = warehouse_id

        try:
            items = get_all_items(warehouse_id)
            if not items:
                optimization_state['running'] = False
                return

            warehouse = get_warehouse_config(warehouse_id)
            ga = GeneticAlgorithm(population_size, generations)
            solution, fitness = ga.optimize(items, warehouse, weights, callback)

            finalize_optimization(solution, 'GA', weights, optimization_state['start_time'], warehouse_id)

        except Exception as e:
            print(f"Optimization error: {e}")
        finally:
            optimization_state['running'] = False
            optimization_state['start_time'] = None

    thread = threading.Thread(target=run_optimization)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'GA optimization started'})


@app.route('/api/optimize/eo', methods=['POST'])
def optimize_eo():
    data = request.json
    iterations = data.get('iterations', 200)
    weights = data.get('weights', {})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    def callback(progress, current_fitness, best_fitness, best_solution, space, access, stability):
        optimization_state['progress'] = progress
        optimization_state['current_fitness'] = current_fitness
        optimization_state['best_fitness'] = best_fitness
        optimization_state['best_solution'] = best_solution

    def run_optimization():
        optimization_state['running'] = True
        optimization_state['algorithm'] = 'EO'
        optimization_state['progress'] = 0
        optimization_state['start_time'] = time.time()
        optimization_state['current_warehouse_id'] = warehouse_id

        try:
            items = get_all_items(warehouse_id)
            if not items:
                optimization_state['running'] = False
                return

            warehouse = get_warehouse_config(warehouse_id)
            eo = ExtremalOptimization(iterations)
            solution, fitness = eo.optimize(items, warehouse, weights, callback)

            finalize_optimization(solution, 'EO', weights, optimization_state['start_time'], warehouse_id)

        except Exception as e:
            print(f"EO optimization error: {e}")
        finally:
            optimization_state['running'] = False
            optimization_state['start_time'] = None

    thread = threading.Thread(target=run_optimization)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'EO optimization started'})


@app.route('/api/optimize/ga-eo', methods=['POST'])
def optimize_ga_eo():
    data = request.json
    ga_generations = data.get('generations', 50)
    eo_iterations = data.get('iterations', 100)
    weights = data.get('weights', {})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    def callback(progress, current_fitness, best_fitness, best_solution, space, access, stability):
        optimization_state['progress'] = progress
        optimization_state['current_fitness'] = current_fitness
        optimization_state['best_fitness'] = best_fitness
        optimization_state['best_solution'] = best_solution

    def run_optimization():
        optimization_state['running'] = True
        optimization_state['algorithm'] = 'GA-EO'
        optimization_state['progress'] = 0
        optimization_state['start_time'] = time.time()
        optimization_state['current_warehouse_id'] = warehouse_id

        try:
            items = get_all_items(warehouse_id)
            if not items:
                optimization_state['running'] = False
                return

            warehouse = get_warehouse_config(warehouse_id)
            solution, fitness = HybridOptimizer.ga_then_eo(
                items, warehouse, ga_generations, eo_iterations, weights, callback)

            finalize_optimization(solution, 'GA-EO', weights, optimization_state['start_time'], warehouse_id)

        except Exception as e:
            print(f"Optimization error: {e}")
        finally:
            optimization_state['running'] = False
            optimization_state['start_time'] = None

    thread = threading.Thread(target=run_optimization)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'GA-EO optimization started'})


@app.route('/api/optimize/eo-ga', methods=['POST'])
def optimize_eo_ga():
    data = request.json
    eo_iterations = data.get('iterations', 100)
    ga_generations = data.get('generations', 50)
    weights = data.get('weights', {})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])

    def callback(progress, current_fitness, best_fitness, best_solution, space, access, stability):
        optimization_state['progress'] = progress
        optimization_state['current_fitness'] = current_fitness
        optimization_state['best_fitness'] = best_fitness
        optimization_state['best_solution'] = best_solution

    def run_optimization():
        optimization_state['running'] = True
        optimization_state['algorithm'] = 'EO-GA'
        optimization_state['progress'] = 0
        optimization_state['start_time'] = time.time()
        optimization_state['current_warehouse_id'] = warehouse_id

        try:
            items = get_all_items(warehouse_id)
            if not items:
                optimization_state['running'] = False
                return

            warehouse = get_warehouse_config(warehouse_id)
            solution, fitness = HybridOptimizer.eo_then_ga(
                items, warehouse, eo_iterations, ga_generations, weights, callback)

            finalize_optimization(solution, 'EO-GA', weights, optimization_state['start_time'], warehouse_id)

        except Exception as e:
            print(f"Optimization error: {e}")
        finally:
            optimization_state['running'] = False
            optimization_state['start_time'] = None

    thread = threading.Thread(target=run_optimization)
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'message': 'EO-GA optimization started'})


@app.route('/api/optimize/compare', methods=['POST'])
def optimize_compare():
    algorithms = [
        {'name': 'GA', 'endpoint': '/api/optimize/ga', 'params': {'population_size': 20, 'generations': 50}},
        {'name': 'EO', 'endpoint': '/api/optimize/eo', 'params': {'iterations': 100}},
    ]
    data = request.json
    weights = data.get('weights', {})
    warehouse_id = data.get('warehouse_id', optimization_state['current_warehouse_id'])
    results = {}

    def run_single_optimization(algo):
        try:
            items = get_all_items(warehouse_id)
            warehouse = get_warehouse_config(warehouse_id)
            start_time = time.time()

            if algo['name'] == 'GA':
                optimizer = GeneticAlgorithm(
                    population_size=algo['params']['population_size'],
                    generations=algo['params']['generations']
                )
                solution, fitness = optimizer.optimize(items, warehouse, weights)
            elif algo['name'] == 'EO':
                optimizer = ExtremalOptimization(iterations=algo['params']['iterations'])
                solution, fitness = optimizer.optimize(items, warehouse, weights)
            else:
                solution, fitness = None, 0

            finalize_optimization(solution, algo['name'], weights, start_time, warehouse_id)

        except Exception as e:
            print(f"Error in {algo['name']}: {e}")
            results[algo['name']] = {'error': str(e)}

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
    warehouse_id = data.get('id', 1)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    levels = data.get('levels', 1)
    height = data.get('height', 1)
    default_layer_heights = [i * (height / levels) for i in range(levels)] if levels > 0 else [0]
    layer_heights = data.get('layer_heights', default_layer_heights)
    layer_heights_json = json.dumps(layer_heights)

    try:
        c.execute('''UPDATE warehouse_config SET 
                     name = ?, length = ?, width = ?, height = ?, grid_size = ?, 
                     levels = ?, walkway_width = ?, layer_heights_json = ? WHERE id = ?''',
                  (data.get('name', 'Warehouse'), data['length'], data['width'], data['height'],
                   data.get('grid_size', 0.5), data.get('levels', 1),
                   data.get('walkway_width', 1.0), layer_heights_json, warehouse_id))

        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/warehouse/zones', methods=['GET'])
def get_exclusion_zones_api():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    zones = get_exclusion_zones(warehouse_id)
    return jsonify(zones)


@app.route('/api/warehouse/zones', methods=['POST'])
def add_exclusion_zone_api():
    data = request.json
    warehouse_id = data.get('warehouse_id', 1)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        c.execute('''INSERT INTO exclusion_zones (name, x1, y1, x2, y2, zone_type, warehouse_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (data['name'], data['x1'], data['y1'], data['x2'], data['y2'],
                   data.get('zone_type', 'exclusion'), warehouse_id))

        conn.commit()
        return jsonify({'success': True, 'id': c.lastrowid})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/warehouse/zones/<zone_id>', methods=['DELETE'])
def delete_exclusion_zone(zone_id):
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()

    try:
        c.execute('DELETE FROM exclusion_zones WHERE id = ? AND warehouse_id = ?', (zone_id, warehouse_id))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


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

    return jsonify({
        'space_utilization': space_util,
        'accessibility': accessibility,
        'stability': stability,
        'grouping': grouping,
        'total_items': len(items),
        'warehouse_volume': warehouse['length'] * warehouse['width'] * warehouse['height']
    })


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    warehouse_id = request.args.get('warehouse_id', default=1, type=int)
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute('''SELECT algorithm, fitness, space_utilization, accessibility, 
                 stability, execution_time, timestamp FROM optimization_results 
                 WHERE warehouse_id = ? ORDER BY timestamp DESC LIMIT 50''', (warehouse_id,))
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            'algorithm': row[0],
            'fitness': row[1],
            'space_utilization': row[2],
            'accessibility': row[3],
            'stability': row[4],
            'execution_time': row[5],
            'timestamp': row[6]
        })

    return jsonify(history)


def migrate_db():
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    try:
        c.execute('SELECT layer_heights_json FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding layer_heights_json column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN layer_heights_json TEXT')
        c.execute("UPDATE warehouse_config SET layer_heights_json = '[]' WHERE layer_heights_json IS NULL")
        conn.commit()

    try:
        c.execute('SELECT grouping FROM optimization_results LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating optimization_results table: adding grouping column.")
        c.execute('ALTER TABLE optimization_results ADD COLUMN grouping REAL DEFAULT 0')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM items LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating items table: adding warehouse_id column.")
        c.execute('ALTER TABLE items ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM exclusion_zones LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating exclusion_zones table: adding warehouse_id column.")
        c.execute('ALTER TABLE exclusion_zones ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT warehouse_id FROM optimization_results LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating optimization_results table: adding warehouse_id column.")
        c.execute('ALTER TABLE optimization_results ADD COLUMN warehouse_id INTEGER DEFAULT 1')
        conn.commit()

    try:
        c.execute('SELECT name FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding name column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN name TEXT')
        c.execute("UPDATE warehouse_config SET name = 'Default Warehouse' WHERE name IS NULL")
        conn.commit()

    try:
        c.execute('SELECT is_active FROM warehouse_config LIMIT 1')
    except sqlite3.OperationalError:
        print("Migrating warehouse_config table: adding is_active column.")
        c.execute('ALTER TABLE warehouse_config ADD COLUMN is_active INTEGER DEFAULT 1')
        conn.commit()

    conn.close()


if __name__ == '__main__':
    init_db()
    migrate_db()
    conn = sqlite3.connect('warehouse.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) from items")
    item_count = c.fetchone()[0]
    conn.close()
    if item_count == 0:
        print("No items found in the database. Loading sample data...")
        load_sample_data()

    app.run(debug=True, host='0.0.0.0', port=5000)