// Global variables
let scene, camera, renderer, controls;
let warehouseGroup, itemsGroup, exclusionZonesGroup, pickerPathGroup;
let coordinateLabels = [];
let warehouseConfig = {};
let allItemsData = {};

let isOptimizing = false;
let optimizationInterval;
let historyChart, categoryChart;
let currentWarehouseId = 1;
let warehouses = [];

// API Base URL
const API_BASE_URL = window.location.origin; // Use current origin

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    loadWarehouses();
    loadWarehouseConfig(); // Initial load
    loadZones(); // Ensure zones are loaded on startup
    switchView('controls');
});

// --- UI Navigation ---
function switchView(viewName) {
    // Hide all view panels
    document.querySelectorAll('.view-panel').forEach(el => el.style.display = 'none');
    // Show selected
    document.getElementById(`view-${viewName}`).style.display = 'block';

    // Update active nav state
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    // Simple mapping based on click handling, or add IDs to nav-items
    // For now assuming the click handler works, but let's visually update if possible
    // (This part relies on the onclick adding 'active' class, which we can do manually here if we had IDs)

    if (viewName === 'analytics') loadAnalytics();
    if (viewName === 'items') loadItemsList();
    if (viewName === 'warehouse') loadWarehouseConfig();
}

// --- Three.js & Visualization ---
function initThreeJS() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050505); // Match CSS --bg-main

    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(20, 20, 20);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 30, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    // Groups
    warehouseGroup = new THREE.Group();
    itemsGroup = new THREE.Group();
    exclusionZonesGroup = new THREE.Group();
    pickerPathGroup = new THREE.Group();
    scene.add(warehouseGroup);
    scene.add(itemsGroup);
    scene.add(exclusionZonesGroup);
    scene.add(pickerPathGroup);

    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('click', onMouseClick);

    animate();
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    if (!container) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Raycasting for Tooltip
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(itemsGroup.children);

    const tooltip = document.getElementById('tooltip');

    if (intersects.length > 0) {
        const data = intersects[0].object.userData;
        tooltip.style.display = 'block';
        tooltip.style.left = (event.clientX + 15) + 'px';
        tooltip.style.top = (event.clientY + 15) + 'px';
        tooltip.innerHTML = `
            <strong>${data.name || 'Item ' + data.id}</strong>
            <div style="font-size:0.8rem; color:#bbb;">
                ${data.width}x${data.height}x${data.length}m<br>
                Weight: ${data.weight}kg<br>
                Pos: ${data.x.toFixed(2)}, ${data.y.toFixed(2)}, ${data.z.toFixed(2)}
            </div>
        `;
        // Select logic could go here (highlighting)
    } else {
        tooltip.style.display = 'none';
    }
}

// --- Data Loading & Rendering ---

function loadWarehouses() {
    fetch(`${API_BASE_URL}/api/warehouses`)
        .then(res => res.json())
        .then(data => {
            warehouses = data;
            const select = document.getElementById('warehouse-select');
            select.innerHTML = '';
            data.forEach(w => {
                const opt = document.createElement('option');
                opt.value = w.id;
                opt.textContent = w.name;
                select.appendChild(opt);
            });
            if (data.length > 0) {
                currentWarehouseId = data[0].id;
                loadWarehouseConfig();
            }
        });
}

function switchWarehouse(id) {
    currentWarehouseId = parseInt(id);
    fetch(`${API_BASE_URL}/api/warehouses/switch/${id}`, { method: 'POST' })
        .then(() => {
            loadWarehouseConfig();
            updateVisualization();
        });
}

function loadWarehouseConfig() {
    fetch(`${API_BASE_URL}/api/warehouse/config?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(config => {
            warehouseConfig = config;
            renderWarehouse();
            updateVisualization(); // Reload items

            // Populate config inputs if view is visible
            document.getElementById('warehouse-length').value = config.length;
            document.getElementById('warehouse-width').value = config.width;
            document.getElementById('warehouse-height').value = config.height;
            document.getElementById('warehouse-grid-size').value = config.grid_size || 1;
            document.getElementById('warehouse-levels').value = config.levels || 1;
            document.getElementById('warehouse-door-x').value = config.door_x || 0;
            document.getElementById('warehouse-door-y').value = config.door_y || 0;
            loadZones();
        });
}

// Helper for Rectangular Grids (Global)
function createRectangularGrid(L, W, step, color, opacity = 0.2, depthTest = true) {
    const points = [];
    const hL = L / 2;
    const hW = W / 2;
    // Ensure step is valid to prevent infinite loops
    if (step <= 0) step = 1;

    // X lines
    for (let z = -hW; z <= hW; z += step) {
        points.push(new THREE.Vector3(-hL, 0, z));
        points.push(new THREE.Vector3(hL, 0, z));
    }
    // Z lines
    for (let x = -hL; x <= hL; x += step) {
        points.push(new THREE.Vector3(x, 0, -hW));
        points.push(new THREE.Vector3(x, 0, hW));
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: color,
        transparent: true,
        opacity: opacity,
        depthTest: depthTest
    });
    return new THREE.LineSegments(geometry, material);
}

function renderWarehouse() {
    // Clear old
    while (warehouseGroup.children.length > 0) warehouseGroup.remove(warehouseGroup.children[0]);

    const { length, width, height, grid_size } = warehouseConfig;

    // Helper for Rectangular Grids


    // Grid Floor
    const gridHelper = createRectangularGrid(length, width, grid_size, 0x333333, 0.3);
    gridHelper.position.y = 0.01;
    warehouseGroup.add(gridHelper);

    // Floor Plane
    const planeGeo = new THREE.PlaneGeometry(length, width);
    const planeMat = new THREE.MeshStandardMaterial({
        color: 0x0a0a0a, side: THREE.DoubleSide, roughness: 0.8
    });
    const floor = new THREE.Mesh(planeGeo, planeMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    warehouseGroup.add(floor);

    // Wireframe Box for Bounds
    const boxGeo = new THREE.BoxGeometry(length, height, width);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x00F0FF, opacity: 0.3, transparent: true }));
    line.position.y = height / 2;
    warehouseGroup.add(line);

    // Render Door
    const doorX = warehouseConfig.door_x || 0;
    const doorY = warehouseConfig.door_y || 0;

    // Draw Door as a distinctive marker (e.g., Neon Box)
    const doorGeo = new THREE.BoxGeometry(2, 0.1, 2); // 2x2m pad
    const doorMat = new THREE.MeshBasicMaterial({ color: 0xFFD600 }); // Yellow
    const doorMesh = new THREE.Mesh(doorGeo, doorMat);

    // Position relative to warehouse center
    doorMesh.position.set(
        doorX - length / 2,
        0.05,
        doorY - width / 2
    );
    warehouseGroup.add(doorMesh);

    // Door Label (Simple vertical stick)
    const stickGeo = new THREE.CylinderGeometry(0.1, 0.1, 2, 8);
    const stickMat = new THREE.MeshBasicMaterial({ color: 0xFFD600 });
    const stick = new THREE.Mesh(stickGeo, stickMat);
    stick.position.set(doorX - length / 2, 1, doorY - width / 2);
    warehouseGroup.add(stick);

    // Render Layer Planes - REMOVED per user request (only show grids in zones)
    /*
    let layers = [];
    if (warehouseConfig.layer_heights && warehouseConfig.layer_heights.length > 0) {
        layers = warehouseConfig.layer_heights;
    } else {
        const lvls = warehouseConfig.levels || 1;
        const hPerLvl = height / lvls;
        for (let i = 0; i < lvls; i++) layers.push(hPerLvl);
    }

    let currentH = 0;
    // Skip ground floor (already drawn)
    for (let i = 0; i < layers.length; i++) {
        currentH += layers[i];
        if (currentH >= height) break; // Don't draw ceiling if it's top

        // Code removed to prevent global grid rendering
    }
    */

    renderGridLabels(length, width);
}

function renderGridLabels(length, width) {
    // Clear old labels
    coordinateLabels.forEach(l => scene.remove(l));
    coordinateLabels = [];

    function createLabel(text, pos) {
        const canvas = document.createElement('canvas');
        canvas.width = 128; // Increased
        canvas.height = 64; // Increased
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgba(0,0,0,0)'; // Transparent
        ctx.fillRect(0, 0, 128, 64);
        ctx.fillStyle = '#00F0FF';
        ctx.font = 'bold 24px monospace'; // Bigger
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 64, 32); // Centered

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.position.copy(pos);
        sprite.scale.set(2, 1, 1);

        scene.add(sprite);
        coordinateLabels.push(sprite);
    }

    // X Axis Labels (Along Back Edge: Z = -width/2)
    for (let x = 0; x <= length; x += 2) {
        createLabel(x.toString(), new THREE.Vector3(x - length / 2, 0.5, -width / 2 - 1));
    }

    // Z Axis (Width) Labels (Along Left Edge: X = -length/2)
    for (let z = 0; z <= width; z += 2) {
        createLabel(z.toString(), new THREE.Vector3(-length / 2 - 1, 0.5, z - width / 2));
    }

    // Y Axis (Height) Labels (At Back-Left Corner)
    // Use warehouseConfig.height or just go up to reasonable amount
    const h = warehouseConfig.height || 5;
    for (let y = 0; y <= h; y += 1) { // Every 1m
        createLabel(y.toString() + 'm', new THREE.Vector3(-length / 2 - 1, y, -width / 2 - 1));
    }

}

function updateVisualization() {
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            items.forEach(i => allItemsData[i.id] = i);
            renderItems(items);
        });
}

function renderItems(items) {
    while (itemsGroup.children.length > 0) itemsGroup.remove(itemsGroup.children[0]);
    if (pickerPathGroup) {
        while (pickerPathGroup.children.length > 0) pickerPathGroup.remove(pickerPathGroup.children[0]);
    }

    const mode = document.getElementById('color-mode') ? document.getElementById('color-mode').value : 'category';
    const showPath = document.getElementById('show-path') ? document.getElementById('show-path').checked : false;

    // Calculate stats for heatmaps
    let maxWeight = 0;
    let maxAccess = 0;
    items.forEach(i => {
        if (i.weight > maxWeight) maxWeight = i.weight;
        if (i.access_freq > maxAccess) maxAccess = i.access_freq;
    });

    if (showPath) {
        renderPickerPath(items);
    }

    items.forEach(item => {
        const geometry = new THREE.BoxGeometry(item.length, item.height, item.width);
        const color = getItemColor(item, mode, maxWeight, maxAccess);
        const material = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.3,
            metalness: 0.1
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(
            item.x - warehouseConfig.length / 2, // Adjust for center origin
            item.z + item.height / 2,
            item.y - warehouseConfig.width / 2
        );
        mesh.rotation.y = -item.rotation * (Math.PI / 180); // Database stores degrees

        mesh.userData = item;
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        itemsGroup.add(mesh);
    });
}

function renderPickerPath(items) {
    if (!items || items.length === 0) return;

    // Greedy TSP to simulation picking order
    const doorX = (warehouseConfig.door_x || 0);
    const doorY = (warehouseConfig.door_y || 0);
    const L = warehouseConfig.length;
    const W = warehouseConfig.width;

    let currentPos = { x: doorX, y: doorY };
    let remaining = [...items].map(i => ({
        ...i,
        // Calculate center for distance
        cx: i.x,
        cy: i.y
    }));

    const points = [];
    // Start at door
    points.push(new THREE.Vector3(doorX - L / 2, 0.1, doorY - W / 2));

    while (remaining.length > 0) {
        let nearestIdx = -1;
        let minDst = Infinity;

        for (let i = 0; i < remaining.length; i++) {
            const item = remaining[i];
            const dst = Math.sqrt(Math.pow(item.cx - currentPos.x, 2) + Math.pow(item.cy - currentPos.y, 2));
            if (dst < minDst) {
                minDst = dst;
                nearestIdx = i;
            }
        }

        if (nearestIdx !== -1) {
            const nextItem = remaining[nearestIdx];
            // Add point
            points.push(new THREE.Vector3(nextItem.cx - L / 2, 0.5, nextItem.cy - W / 2)); // Path slightly elevated
            currentPos = { x: nextItem.cx, y: nextItem.cy };
            remaining.splice(nearestIdx, 1);
        } else {
            break;
        }
    }

    // Return to door? Optional. For now just path to last item.

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color: 0xFFFF00, linewidth: 2 });
    const line = new THREE.Line(geometry, material);
    pickerPathGroup.add(line);
}

function getItemColor(item, mode, maxWeight, maxAccess) {
    if (mode === 'category') {
        const colors = {
            'Electronics': 0x00F0FF,
            'Furniture': 0x7000FF,
            'Fragile': 0xFF0055,
            'Heavy': 0xFFD600,
            'General': 0xAAAAAA
        };
        return colors[item.category] || 0x888888;
    } else if (mode === 'weight') {
        // Green (light) -> Red (heavy)
        const t = maxWeight > 0 ? (item.weight / maxWeight) : 0;
        const color = new THREE.Color().setHSL(0.33 - (t * 0.33), 1, 0.5); // 0.33=Green, 0=Red
        return color;
    } else if (mode === 'access') {
        // Blue (low) -> Red (high)
        const t = maxAccess > 0 ? (item.access_freq / maxAccess) : 0;
        const color = new THREE.Color().setHSL(0.66 - (t * 0.66), 1, 0.5); // 0.66=Blue, 0=Red
        return color;
    }
    return 0x888888;
}

// --- Optimization Controls ---

function startOptimization() {
    const algo = document.getElementById('algorithm-select').value;

    const params = {
        warehouse_id: currentWarehouseId,
        population_size: parseInt(document.getElementById('population-size').value),
        generations: parseInt(document.getElementById('generations').value),
        weights: {
            space: parseFloat(document.getElementById('weight-space').value),
            accessibility: parseFloat(document.getElementById('weight-accessibility').value),
            stability: parseFloat(document.getElementById('weight-stability').value)
        }
    };

    // Ensure allItemsData is populated before starting optimization
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            // Populate allItemsData to ensure 3D view can render during optimization
            items.forEach(i => allItemsData[i.id] = i);

            // Now start the optimization
            startOptimizationRequest(algo, params);
        })
        .catch(err => {
            alert(`Failed to load items: ${err}`);
        });
}

function startOptimizationRequest(algo, params) {

    // Fix API endpoint logic based on algo
    let endpoint = `/api/optimize/${algo}`;
    if (algo.includes('hybrid')) endpoint = '/api/optimize/ga-eo'; // Simplified for now

    fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                isOptimizing = true;
                const startBtn = document.getElementById('start-btn');
                const stopBtn = document.getElementById('stop-btn');
                const statusText = document.getElementById('status-text');
                const statusDot = document.getElementById('status-dot');

                if (startBtn) startBtn.disabled = true;
                if (stopBtn) stopBtn.disabled = false;
                if (statusText) statusText.textContent = "OPTIMIZING...";
                if (statusDot) statusDot.classList.add('active');

                startPolling();
            } else {
                alert(`Error starting optimization: ${data.error}`);
            }
        })
        .catch(err => {
            alert(`Network error: ${err}`);
        });
}

function stopOptimization() {
    fetch(`${API_BASE_URL}/api/optimize/stop`, { method: 'POST' })
        .then(() => {
            isOptimizing = false;
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
        });
}

function startPolling() {
    optimizationInterval = setInterval(() => {
        fetch(`${API_BASE_URL}/api/optimize/status`)
            .then(res => res.json())
            .then(status => {
                if (!status.running) {
                    clearInterval(optimizationInterval);
                    isOptimizing = false;

                    const startBtn = document.getElementById('start-btn');
                    const stopBtn = document.getElementById('stop-btn');
                    if (startBtn) startBtn.disabled = false;
                    if (stopBtn) stopBtn.disabled = true;

                    const statusText = document.getElementById('status-text');
                    const statusDot = document.getElementById('status-dot');
                    if (statusText) statusText.textContent = "COMPLETED";
                    if (statusDot) statusDot.classList.remove('active');
                    loadAnalytics(); // Refresh stats
                    updateVisualization(); // Reload items from DB to ensure view matches final result
                }

                // Update UI
                const progress = status.progress || 0;
                const progPct = document.getElementById('progress-percent');
                const progBar = document.getElementById('progress-bar-fill');
                const bestFit = document.getElementById('best-fitness');

                if (progPct) progPct.textContent = Math.round(progress) + '%';
                if (progBar) progBar.style.width = progress + '%';
                if (bestFit) bestFit.textContent = (status.best_fitness || 0).toFixed(4);

                // Update Status Text with detailed message
                const statusText = document.getElementById('status-text');
                if (statusText) {
                    if (status.message) {
                        statusText.textContent = status.message;
                        // Optional: Add tooltip or title for very long messages
                        statusText.title = status.message;
                    } else {
                        statusText.textContent = "OPTIMIZING...";
                    }
                }


                if (status.best_solution && status.best_solution.length > 0) {
                    console.log('[DEBUG] best_solution items:', status.best_solution.length);
                    console.log('[DEBUG] allItemsData keys:', Object.keys(allItemsData).length);
                    // Map solution coordinates back to full item data
                    const solutionItems = status.best_solution.map(sol => {
                        const originalItem = allItemsData[sol.id];
                        if (!originalItem) {
                            console.log('[DEBUG] Missing item in allItemsData:', sol.id);
                        }
                        if (originalItem) {
                            return { ...originalItem, ...sol };
                        }
                        return null;
                    }).filter(item => item !== null);

                    console.log('[DEBUG] solutionItems after mapping:', solutionItems.length);

                    // Only re-render if we have items to show (prevents blank screen)
                    if (solutionItems.length > 0) {
                        renderItems(solutionItems);
                    }
                }
            });
    }, 200); // Faster polling for smooth updates
}

// --- Analytics ---
function loadAnalytics() {
    fetch(`${API_BASE_URL}/api/metrics/current?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            const mSpace = document.getElementById('metric-space');
            const mAccess = document.getElementById('metric-access');
            const mStab = document.getElementById('metric-stability');
            const mCount = document.getElementById('metric-count');

            if (mSpace) mSpace.textContent = (data.space_utilization * 100).toFixed(1) + '%';
            if (mAccess) mAccess.textContent = data.accessibility.toFixed(2);
            if (mStab) mStab.textContent = (data.stability * 100).toFixed(1) + '%';
            if (mCount) mCount.textContent = data.total_items;
        });

    fetch(`${API_BASE_URL}/api/metrics/categories?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            initCategoryChart(data);
        });

    // Fetch Algorithm Best Performance
    fetch(`${API_BASE_URL}/api/metrics/algo-best?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            const algoBestBody = document.getElementById('algo-best-body');
            if (algoBestBody) {
                algoBestBody.innerHTML = '';

                if (data.length === 0) {
                    algoBestBody.innerHTML = '<tr><td colspan="4" style="padding:10px; color:var(--text-muted);">No optimization runs yet</td></tr>';
                    return;
                }

                data.forEach(algo => {
                    const row = document.createElement('tr');
                    row.style.borderBottom = '1px solid #333';

                    // Format algorithm name
                    let algoName = algo.algorithm;
                    let algoColor = '#fff';
                    if (algoName.includes('Genetic') && !algoName.includes('Hybrid')) {
                        algoName = 'GA';
                        algoColor = '#00f0ff';
                    } else if (algoName.includes('Extremal') && !algoName.includes('Hybrid')) {
                        algoName = 'EO';
                        algoColor = '#7000ff';
                    } else if (algoName.includes('GA+EO') || algoName.includes('GA-EO')) {
                        algoName = 'GA+EO';
                        algoColor = '#ff0055';
                    } else if (algoName.includes('EO+GA') || algoName.includes('EO-GA')) {
                        algoName = 'EO+GA';
                        algoColor = '#FFD600';
                    }

                    // Format timestamp
                    let achievedAt = '-';
                    if (algo.timestamp) {
                        const date = new Date(algo.timestamp);
                        achievedAt = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    }

                    row.innerHTML = `
                        <td style="padding:8px; font-weight:bold; color:${algoColor};">${algoName}</td>
                        <td style="padding:8px; font-family:'JetBrains Mono'; color:var(--text-main);">${algo.best_fitness ? algo.best_fitness.toFixed(4) : '0.0000'}</td>
                        <td style="padding:8px; color:#00f0ff; font-family:'JetBrains Mono';">${algo.time_to_best ? algo.time_to_best.toFixed(2) + 's' : '-'}</td>
                        <td style="padding:8px; color:var(--text-muted); font-size:0.75rem;">${achievedAt}</td>
                    `;
                    algoBestBody.appendChild(row);
                });
            }
        });

    fetch(`${API_BASE_URL}/api/metrics/history?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(data => {
            initHistoryChart(data);

            // Populate History Log Table
            const historyBody = document.getElementById('analytics-history-body');
            if (historyBody) {
                historyBody.innerHTML = '';
                data.forEach(run => {
                    const row = document.createElement('tr');
                    row.style.borderBottom = '1px solid #333';

                    let algoName = run.algorithm.replace('_COMPARE', ' (C)');
                    if (algoName === 'Genetic Algorithm') algoName = 'GA';
                    if (algoName === 'Extremal Optimization') algoName = 'EO';

                    row.innerHTML = `
                        <td style="padding:6px; font-weight:bold; color:var(--text-main);">${algoName}</td>
                        <td style="padding:6px;">${run.fitness ? run.fitness.toFixed(1) : '0.0'}</td>
                        <td style="padding:6px; color:#00f0ff;">${run.time_to_best ? run.time_to_best.toFixed(2) + 's' : '-'}</td>
                        <td style="padding:6px;">${run.execution_time ? run.execution_time.toFixed(2) + 's' : '-'}</td>
                    `;
                    historyBody.appendChild(row);
                });
            }
        });
}

function initHistoryChart(history) {
    const ctx = document.getElementById('historyChart');
    if (!ctx) return;
    if (historyChart) historyChart.destroy();

    // Map history to chart data
    // Group by timestamp or just show last N runs
    const data = history.slice(-10); // Last 10
    const labels = data.map(d => new Date(d.timestamp).toLocaleTimeString());
    const fitness = data.map(d => d.fitness);

    historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Fitness Score',
                data: fitness,
                borderColor: '#00F0FF',
                backgroundColor: 'rgba(0, 240, 255, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8899A6' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8899A6' }
                }
            },
            plugins: {
                legend: { labels: { color: '#8899A6' } }
            }
        }
    });
}

function initCategoryChart(data) {
    const ctx = document.getElementById('categoryChart');
    if (!ctx) return;

    if (categoryChart) categoryChart.destroy();

    categoryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.categories,
            datasets: [{
                data: data.counts,
                backgroundColor: ['#00F0FF', '#7000FF', '#FF0055', '#FFD600', '#FFFFFF'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#8899A6' } }
            }
        }
    });
}

// --- Item List ---
function loadItemsList() {
    const container = document.getElementById('items-list');
    container.innerHTML = 'Loading...';

    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            container.innerHTML = '';
            const table = document.createElement('table');
            table.style.width = '100%';
            table.style.borderCollapse = 'collapse';

            items.forEach(item => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
                tr.innerHTML = `
                    <td style="padding: 8px;">${item.name || item.id}</td>
                    <td style="padding: 8px; color: var(--accent-primary)">${item.width}x${item.height}x${item.length}</td>
                    <td style="padding: 8px;">${item.category}</td>
                `;
                table.appendChild(tr);
            });
            container.appendChild(table);
        });
}

let editingZoneId = null;
let currentZones = [];

function loadZones() {
    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            currentZones = zones; // Store locally
            renderZones(zones);
            updateZonesList(zones);
        });
}






function editZone(id) {
    const zone = currentZones.find(z => z.id === id);
    if (!zone) return;

    document.getElementById('zone-name').value = zone.name;
    document.getElementById('zone-x1').value = zone.x1;
    document.getElementById('zone-y1').value = zone.y1;
    document.getElementById('zone-x2').value = zone.x2;
    document.getElementById('zone-y2').value = zone.y2;
    document.getElementById('zone-z1').value = zone.z1 || 0;
    document.getElementById('zone-z2').value = zone.z2 || '';
    document.getElementById('zone-type').value = zone.zone_type;

    // Load Metadata
    if (zone.metadata) {
        if (zone.metadata.layer_heights) {
            document.getElementById('zone-layer-heights').value = zone.metadata.layer_heights.join(', ');
        } else {
            document.getElementById('zone-layer-heights').value = '';
        }

        if (zone.metadata.levels) {
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = zone.metadata.levels;
        } else {
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = '';
        }
    } else {
        document.getElementById('zone-layer-heights').value = '';
        const levelInput = document.getElementById('zone-levels');
        if (levelInput) levelInput.value = '';
    }

    editingZoneId = id;
    const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
    if (btn) btn.innerText = "UPDATE ZONE";
}



function renderZones(zones) {
    // Clear old zones
    // Assuming exclusionZonesGroup was defined in initThreeJS
    while (exclusionZonesGroup.children.length > 0) {
        exclusionZonesGroup.remove(exclusionZonesGroup.children[0]);
    }

    zones.forEach(zone => {
        const width = zone.x2 - zone.x1;
        const depth = zone.y2 - zone.y1;

        // Z coordinates
        const whHeight = warehouseConfig ? warehouseConfig.height : 5;
        let z1 = zone.z1 !== undefined ? zone.z1 : 0;
        let z2 = zone.z2 !== undefined ? zone.z2 : whHeight;
        // Clamp z2 if it's the default 100 and likely unset (heuristic)
        if (z2 === 100 && whHeight < 100) z2 = whHeight;

        const zoneHeight = z2 - z1;
        if (zoneHeight <= 0) return; // Skip invalid

        const geometry = new THREE.BoxGeometry(width, zoneHeight, depth);
        const isAlloc = zone.zone_type === 'allocation';
        const color = isAlloc ? 0x00FF9D : 0xFF0055;

        const material = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.3, // Increased from 0.15 for better visibility
            side: THREE.DoubleSide
        });

        const mesh = new THREE.Mesh(geometry, material);

        // Alignment
        const whLength = warehouseConfig ? warehouseConfig.length : 20;
        const whWidth = warehouseConfig ? warehouseConfig.width : 20;

        mesh.position.set(
            (zone.x1 + width / 2) - whLength / 2,
            z1 + zoneHeight / 2,
            (zone.y1 + depth / 2) - whWidth / 2
        );

        // Wireframe
        const edges = new THREE.EdgesGeometry(geometry);
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
            color: color,
            depthTest: false, // Always show on top of other objects
            lights: false
        }));
        line.renderOrder = 1; // Try to force render on top
        mesh.add(line);

        // Shelf Layers
        if (isAlloc) {
            let layerHeights = [];
            if (zone.metadata && zone.metadata.layer_heights && zone.metadata.layer_heights.length > 0) {
                layerHeights = zone.metadata.layer_heights;
            } else {
                // Fallback to levels count
                // DEFAULT TO 1 (Ground only) if not specified in metadata. Ignore global config to prevent unwanted grids.
                const levels = (zone.metadata && zone.metadata.levels) ? parseInt(zone.metadata.levels) : 1;

                if (levels > 1) {
                    const h = zoneHeight / levels;
                    for (let i = 0; i < levels; i++) layerHeights.push(h);
                }
            }

            // Draw planes
            // If levels=3 (Ground, Shelf1, Shelf2), we need dividers at z=H/3 and z=2H/3
            // The bottom (currentY) starts at -zoneHeight/2
            let currentY = -zoneHeight / 2;
            const hPerLayer = zoneHeight / layerHeights.length;

            for (let i = 0; i < layerHeights.length; i++) {
                // Determine height of this layer
                const h = layerHeights[i];
                currentY += h;

                // Don't draw top face if it's the very top of box
                if (currentY >= zoneHeight / 2 - 0.01) break;

                // Add Shelf Plane (Transparent visual)
                const shelfGeo = new THREE.PlaneGeometry(width, depth);
                const shelfMat = new THREE.MeshBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: 0.6, // Increased opacity for visibility
                    side: THREE.DoubleSide
                });
                const shelf = new THREE.Mesh(shelfGeo, shelfMat);
                shelf.rotation.x = Math.PI / 2;
                shelf.position.y = currentY;

                const shelfEdges = new THREE.EdgesGeometry(shelfGeo);
                const shelfLine = new THREE.LineSegments(shelfEdges, new THREE.LineBasicMaterial({ color: 0xFFFFFF }));
                shelf.add(shelfLine);

                mesh.add(shelf);

                // Add Grid to Shelf (White grid lines)
                // Grid is XZ, matches mesh local space. Position at same height as shelf.
                const gridSize = warehouseConfig.grid_size || 1;
                // High opacity (0.9) and depthTest: false to ensure it renders on top of the zone box transparency
                const layerGrid = createRectangularGrid(width, depth, gridSize, 0xFFFFFF, 1.0, false);
                layerGrid.position.y = currentY + 0.01; // Slightly above shelf to avoid z-fighting
                // Ensure renderOrder is later than the box to appear on "top"
                layerGrid.renderOrder = 2; // Box edges are 1
                mesh.add(layerGrid);
            }
        }

        exclusionZonesGroup.add(mesh);
    });
}

function updateZonesList(zones) {
    const container = document.getElementById('zones-list');
    if (!container) return;

    container.innerHTML = zones.map(z => {
        const isAlloc = z.zone_type === 'allocation';
        const layerInfo = (z.metadata && z.metadata.layer_heights && z.metadata.layer_heights.length > 0)
            ? `(Custom Layers)`
            : ((z.metadata && z.metadata.levels)
                ? `(Layers: ${z.metadata.levels})`
                : '');

        return `
        <div class="data-card" style="margin-bottom: 5px; padding: 10px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="color: ${isAlloc ? 'var(--success)' : 'var(--danger)'}">
                    ${z.name} (${isAlloc ? 'Alloc' : 'Restr'})
                </strong>
                <div style="font-size: 0.7rem; color: var(--text-muted);">
                    X: ${z.x1}-${z.x2}, Y: ${z.y1}-${z.y2}<br>
                    Z: ${z.z1 || 0}-${z.z2 || 'Max'} 
                    ${layerInfo}
                </div>
            </div>
            <div>
                 <button class="btn" style="padding: 4px 8px; font-size: 0.7rem; margin-right:5px; background: var(--accent-primary);" onclick="editZone(${z.id})">EDIT</button>
                 <button class="btn btn-danger" style="padding: 4px 8px; font-size: 0.7rem;" onclick="deleteZone(${z.id})">DEL</button>
            </div>
        </div>
    `}).join('');
}

function addZone() {
    const name = document.getElementById('zone-name').value;

    // Read Position & Size Inputs
    const posX = parseFloat(document.getElementById('zone-pos-x').value);
    const posY = parseFloat(document.getElementById('zone-pos-y').value);
    const width = parseFloat(document.getElementById('zone-width').value);
    const depth = parseFloat(document.getElementById('zone-depth').value);

    // Read Vertical Inputs
    const baseZ = parseFloat(document.getElementById('zone-base-z').value) || 0;
    const height = parseFloat(document.getElementById('zone-height').value) || 5;

    // Convert to Backend Format (x1, y1, x2, y2)
    const x1 = posX;
    const y1 = posY;
    const x2 = posX + width;
    const y2 = posY + depth;
    const z1 = baseZ;
    const z2 = baseZ + height;

    const type = document.getElementById('zone-type').value;
    const layerStr = document.getElementById('zone-layer-heights').value;
    const levelsVal = document.getElementById('zone-levels') ? parseInt(document.getElementById('zone-levels').value) : null;

    let layerHeights = [];
    if (layerStr) {
        layerHeights = layerStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    }

    if (!name || isNaN(x1) || isNaN(y1) || isNaN(width) || isNaN(depth)) {
        alert("Please fill name, position, and dimensions");
        return;
    }

    const payload = {
        warehouse_id: currentWarehouseId,
        name: name,
        x1: x1, y1: y1,
        z1: z1, z2: z2,
        x2: x2, y2: y2,
        zone_type: type,
        metadata: {
            layer_heights: layerHeights,
            levels: levelsVal
        }
    };

    let url = `${API_BASE_URL}/api/warehouse/zones`;
    let method = 'POST';

    // Check if we are updating (editingZoneId is set)
    if (editingZoneId) {
        url = `${API_BASE_URL}/api/warehouse/zones/${editingZoneId}`;
        method = 'PUT';
    }

    fetch(url, {
        method: method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                loadZones();
                // Clear inputs
                document.getElementById('zone-name').value = '';
                document.getElementById('zone-pos-x').value = '';
                document.getElementById('zone-pos-y').value = '';
                document.getElementById('zone-width').value = '';
                document.getElementById('zone-depth').value = '';
                document.getElementById('zone-base-z').value = '0';
                document.getElementById('zone-height').value = '5';
                document.getElementById('zone-layer-heights').value = '';
                const levelInput = document.getElementById('zone-levels');
                if (levelInput) levelInput.value = '1';

                // Reset Edit Mode
                editingZoneId = null;
                const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
                if (btn) btn.innerText = "ADD ZONE";
            }
        });
}

function editZone(id) {
    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            const z = zones.find(zn => zn.id === id);
            if (!z) return;

            document.getElementById('zone-name').value = z.name;

            // Convert x1/x2 -> Pos/Size
            const width = z.x2 - z.x1;
            const depth = z.y2 - z.y1;
            const height = (z.z2 || 5) - (z.z1 || 0);

            document.getElementById('zone-pos-x').value = z.x1;
            document.getElementById('zone-pos-y').value = z.y1;
            document.getElementById('zone-width').value = width;
            document.getElementById('zone-depth').value = depth;

            document.getElementById('zone-base-z').value = z.z1 || 0;
            document.getElementById('zone-height').value = height;

            document.getElementById('zone-type').value = z.zone_type;

            const layers = (z.metadata && z.metadata.layer_heights) ? z.metadata.layer_heights.join(', ') : '';
            document.getElementById('zone-layer-heights').value = layers;

            const levels = (z.metadata && z.metadata.levels) ? z.metadata.levels : 1;
            const levelInput = document.getElementById('zone-levels');
            if (levelInput) levelInput.value = levels;

            editingZoneId = id;
            const btn = document.querySelector('#view-warehouse button[onclick="addZone()"]');
            if (btn) btn.innerText = "UPDATE ZONE";
        });
}

function deleteZone(id) {
    if (!confirm('Delete this zone?')) return;
    console.log('Deleting zone:', id);
    fetch(`${API_BASE_URL}/api/warehouse/zones/${id}?warehouse_id=${currentWarehouseId}`, { method: 'DELETE' })
        .then(res => res.json())
        .then(data => {
            console.log('Delete response:', data);
            if (data.success) {
                loadZones();
            } else {
                alert('Failed to delete zone: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(err => {
            console.error('Delete error:', err);
            alert('Error deleting zone: ' + err);
        });
}


function applyPreset(name) {
    if (!confirm('This will overwrite current warehouse configuration and zones. Continue?')) return;

    console.log('Applying preset:', name);

    if (name === '4-shelves') {
        // 1. Update Config
        const newConfig = {
            name: "Warehouse 4-Shelves",
            length: 20,
            width: 15,
            height: 6,
            levels: 2,
            grid_size: 1,
            id: currentWarehouseId
        };

        console.log('Sending config:', newConfig);

        fetch(`${API_BASE_URL}/api/warehouse/config`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        }).then(res => res.json())
            .then(d => {
                console.log('Config response:', d);
                if (d.success) {
                    // 2. Clear Zones & Add New Ones
                    clearZones().then(() => {
                        const zones = [
                            { name: 'Shelf A', x1: 2, y1: 2, x2: 8, y2: 6, zone_type: 'allocation', metadata: { levels: 2 } },
                            { name: 'Shelf B', x1: 12, y1: 2, x2: 18, y2: 6, zone_type: 'allocation', metadata: { levels: 2 } },
                            { name: 'Shelf C', x1: 2, y1: 9, x2: 8, y2: 13, zone_type: 'allocation', metadata: { levels: 2 } },
                            { name: 'Shelf D', x1: 12, y1: 9, x2: 18, y2: 13, zone_type: 'allocation', metadata: { levels: 2 } }
                        ];

                        const promises = zones.map(z =>
                            fetch(`${API_BASE_URL}/api/warehouse/zones`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ ...z, warehouse_id: currentWarehouseId })
                            })
                        );

                        Promise.all(promises).then(() => {
                            loadWarehouseConfig();
                            alert('Preset Applied: 4 Shelves (2 Layers)');
                        });
                    });
                }
            });
    } else if (name === 'clear') {
        clearZones().then(() => {
            // Also reset config to default
            const defaultConfig = {
                name: "Warehouse " + currentWarehouseId,
                length: 20,
                width: 20,
                height: 5,
                levels: 1,
                grid_size: 1,
                door_x: 0,
                door_y: 0,
                id: currentWarehouseId
            };

            fetch(`${API_BASE_URL}/api/warehouse/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(defaultConfig)
            }).then(() => {
                loadWarehouseConfig();
                loadZones();
                alert('Cleared: Zones removed and configuration reset.');
            });
        });
    }
}

function deleteAllItems() {
    if (!confirm('Are you sure you want to delete ALL items? This cannot be undone.')) {
        return;
    }

    fetch(`${API_BASE_URL}/api/items/delete_all?warehouse_id=${currentWarehouseId}`, {
        method: 'DELETE'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                loadItems();
                updateVisualization();
                alert('All items deleted successfully.');
            } else {
                alert('Error deleting items: ' + data.error);
            }
        })
        .catch(error => console.error('Error:', error));
}


function clearZones() {
    console.log('Clearing all zones...');
    return fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            if (!Array.isArray(zones)) {
                console.error('Expected array of zones, got:', zones);
                throw new Error('Invalid response from server when fetching zones.');
            }
            console.log(`Found ${zones.length} zones to clear.`);
            const promises = zones.map(z =>
                fetch(`${API_BASE_URL}/api/warehouse/zones/${z.id}?warehouse_id=${currentWarehouseId}`, { method: 'DELETE' })
            );
            return Promise.all(promises);
        })
        .catch(err => {
            console.error('Error clearing zones:', err);
            alert('Error clearing zones: ' + err.message);
            throw err; // Re-throw to stop chain if needed
        });
}


function updateWarehouseConfig() {
    const data = {
        name: "Warehouse " + currentWarehouseId,
        length: parseFloat(document.getElementById('warehouse-length').value),
        width: parseFloat(document.getElementById('warehouse-width').value),
        height: parseFloat(document.getElementById('warehouse-height').value),
        grid_size: parseFloat(document.getElementById('warehouse-grid-size').value) || 1,
        levels: parseInt(document.getElementById('warehouse-levels').value) || 1,
        door_x: parseFloat(document.getElementById('warehouse-door-x').value) || 0,
        door_y: parseFloat(document.getElementById('warehouse-door-y').value) || 0,
        id: currentWarehouseId
    };

    fetch(`${API_BASE_URL}/api/warehouse/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).then(res => res.json())
        .then(d => {
            if (d.success) {
                loadWarehouseConfig();
                alert('Configuration Saved');
            }
        });
}

function uploadCSV() {
    const input = document.getElementById('file-upload');
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch(`${API_BASE_URL}/api/upload-csv?warehouse_id=${currentWarehouseId}`, {
        method: 'POST',
        body: formData
    }).then(res => res.json())
        .then(d => {
            if (d.success) {
                alert('Uploaded Successfully');
                updateVisualization();
                if (document.getElementById('view-items').style.display === 'block') loadItemsList();
            } else {
                alert('Error: ' + d.error);
            }
        });
}

function runComparison() {
    const weights = {
        space: parseFloat(document.getElementById('weight-space').value),
        accessibility: parseFloat(document.getElementById('weight-accessibility').value),
        stability: parseFloat(document.getElementById('weight-stability').value)
    };

    const modal = document.getElementById('comparison-modal');
    const content = document.getElementById('comparison-results-content');
    modal.style.display = 'flex';
    content.innerHTML = '<div style="text-align:center;">Running optimizations... please wait.<br><div class="dot active" style="margin:20px auto;"></div></div>';

    fetch(`${API_BASE_URL}/api/optimize/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            warehouse_id: currentWarehouseId,
            weights: weights
        })
    }).then(res => res.json())
        .then(data => {
            if (data.success) {
                let html = '<table style="width:100%; border-collapse:collapse; text-align:left;">';
                html += '<thead><tr style="border-bottom:1px solid #444; color:var(--accent-primary);">';
                html += '<th style="padding:10px;">Metric</th>';
                Object.keys(data.results).forEach(algo => {
                    html += `<th style="padding:10px;">${algo}</th>`;
                });
                html += '</tr></thead><tbody>';

                const metrics = [
                    { key: 'fitness', label: 'Total Fitness' },
                    { key: 'time', label: 'Time (s)' },
                    { key: 'space_utilization', label: 'Space Util' },
                    { key: 'accessibility', label: 'Accessibility' },
                    { key: 'stability', label: 'Stability' }
                ];

                metrics.forEach(m => {
                    html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">`;
                    html += `<td style="padding:10px; color:#fff;">${m.label}</td>`;
                    Object.keys(data.results).forEach(algo => {
                        let val = data.results[algo][m.key];
                        if (typeof val === 'number') {
                            if (m.key === 'time') val = val.toFixed(4) + 's';
                            else if (m.key === 'fitness') val = val.toFixed(4);
                            else if (m.key === 'accessibility') val = val.toFixed(4);
                            else val = (val * 100).toFixed(1) + '%';
                        }

                        // Highlight winner (simple logic)
                        // This logic could be more robust, but good enough for now
                        let color = '#8899A6';
                        html += `<td style="padding:10px; color:${color};">${val}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody></table>';

                content.innerHTML = html;
            } else {
                content.innerHTML = 'Error: ' + JSON.stringify(data);
            }
        })
        .catch(err => {
            content.innerHTML = 'Error: ' + err;
        });
}

// --- Benchmark Functions ---
let benchmarkInterval = null;

function runBenchmark() {
    if (!confirm('This will run each algorithm 20 times. This may take several minutes. Continue?')) {
        return;
    }

    const benchmarkBtn = document.getElementById('benchmark-btn');
    const statusDiv = document.getElementById('benchmark-status');

    if (benchmarkBtn) benchmarkBtn.disabled = true;
    if (statusDiv) statusDiv.style.display = 'block';

    const params = {
        warehouse_id: currentWarehouseId,
        runs: 20,
        generations: 50,  // Reduced for faster benchmarking
        iterations: 500,
        population_size: 30,
        weights: {
            space: parseFloat(document.getElementById('weight-space')?.value || 0.6),
            accessibility: parseFloat(document.getElementById('weight-accessibility')?.value || 0.3),
            stability: parseFloat(document.getElementById('weight-stability')?.value || 0.1)
        }
    };

    fetch(`${API_BASE_URL}/api/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                startBenchmarkPolling();
            } else {
                alert('Error starting benchmark: ' + data.error);
                if (benchmarkBtn) benchmarkBtn.disabled = false;
                if (statusDiv) statusDiv.style.display = 'none';
            }
        })
        .catch(err => {
            alert('Network error: ' + err);
            if (benchmarkBtn) benchmarkBtn.disabled = false;
            if (statusDiv) statusDiv.style.display = 'none';
        });
}

function startBenchmarkPolling() {
    benchmarkInterval = setInterval(() => {
        fetch(`${API_BASE_URL}/api/benchmark/status`)
            .then(res => res.json())
            .then(status => {
                const algoSpan = document.getElementById('benchmark-algo');
                const progressSpan = document.getElementById('benchmark-progress');
                const progressBar = document.getElementById('benchmark-bar');

                // Fix: 4 algorithms now (GA, EO, GA-EO, EO-GA)
                const runsPerAlgo = Math.ceil(status.total_runs / 4);
                if (algoSpan) algoSpan.textContent = `${status.current_algo || '-'} (Run ${status.current_run}/${runsPerAlgo})`;
                if (progressSpan) progressSpan.textContent = Math.round(status.progress) + '%';
                if (progressBar) progressBar.style.width = status.progress + '%';

                // Show real-time results as each algorithm completes
                const resultsDiv = document.getElementById('benchmark-results');
                if (resultsDiv && status.results && Object.keys(status.results).length > 0) {
                    let html = '<div style="margin-top: 10px; font-size: 0.8rem;">';
                    html += '<div style="color: var(--text-muted); margin-bottom: 5px;">Completed:</div>';
                    for (const [key, result] of Object.entries(status.results)) {
                        const color = key === 'GA' ? '#00f0ff' : key === 'EO' ? '#7000ff' : key === 'GA-EO' ? '#ff0055' : '#ffd600';
                        html += `<div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #333;">`;
                        html += `<span style="color: ${color}; font-weight: bold;">${key}</span>`;
                        html += `<span style="font-family: 'JetBrains Mono';">${result.avg_fitness.toFixed(4)} (${result.runs} runs)</span>`;
                        html += `</div>`;
                    }
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                }

                if (!status.running) {
                    clearInterval(benchmarkInterval);
                    benchmarkInterval = null;

                    const benchmarkBtn = document.getElementById('benchmark-btn');
                    const statusDiv = document.getElementById('benchmark-status');

                    if (benchmarkBtn) benchmarkBtn.disabled = false;
                    if (statusDiv) statusDiv.style.display = 'none';

                    // Refresh analytics to show new averaged results
                    loadAnalytics();
                    alert('Benchmark complete! Results have been saved.');
                }
            });
    }, 500);
}

function clearAlgoPerformance() {
    if (!confirm('This will delete all optimization results for this warehouse. Are you sure?')) {
        return;
    }

    fetch(`${API_BASE_URL}/api/metrics/algo-best/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ warehouse_id: currentWarehouseId })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Cleared ${data.deleted} optimization records.`);
                loadAnalytics(); // Refresh the table
            } else {
                alert('Error clearing performance: ' + data.error);
            }
        })
        .catch(err => {
            alert('Network error: ' + err);
        });
}