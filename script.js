// Global variables
let scene, camera, renderer, controls;
let warehouseGroup, itemsGroup, exclusionZonesGroup;
let coordinateLabels = [];
let warehouseConfig = {};
let allItemsData = {}; // Cache for full item data

let isOptimizing = false;
let optimizationInterval;
let lastRunningTime = "00:00";
let historyChart;
let currentWarehouseId = 1;
let warehouses = [];

// Three.js initialization
function initThreeJS() {
    const container = document.getElementById('three-container');

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0D1117);

    // Camera
    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(15, 10, 15);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Warehouse group
    warehouseGroup = new THREE.Group();
    scene.add(warehouseGroup);

    // Items group
    itemsGroup = new THREE.Group();
    scene.add(itemsGroup);

    // Exclusion zones group
    exclusionZonesGroup = new THREE.Group();
    scene.add(exclusionZonesGroup);

    // Initial render
    renderWarehouse();
    loadItemsForVisualization();
    renderExclusionZones();

    // Animation loop
    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Dynamically adjust label size based on camera distance for readability
    if (coordinateLabels && coordinateLabels.length > 0) {
        const baseScaleFactor = 0.05;
        coordinateLabels.forEach(label => {
            if (!label || !label.position) return;
            const distance = camera.position.distanceTo(label.position);
            const scale = distance * baseScaleFactor;

            const aspect = label.userData.aspect || 1;
            label.scale.set(aspect * scale, scale, 1.0);

            label.visible = distance < 75;
        });
    }

    renderer.render(scene, camera);
}

// Helper function to create text labels for grid coordinates
function createCoordinateLabel(text, position, fontSize = 48) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = `Bold ${fontSize}px Inter, sans-serif`;

    const textMetrics = context.measureText(text);
    const textWidth = textMetrics.width;
    const padding = 10;
    canvas.width = textWidth + padding;
    canvas.height = fontSize + padding;

    context.font = `Bold ${fontSize}px Inter, sans-serif`;
    context.fillStyle = 'rgba(139, 148, 158, 0.8)';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;

    const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(material);

    sprite.position.copy(position);

    const aspect = canvas.width / canvas.height;
    sprite.userData.aspect = aspect;

    sprite.scale.set(aspect, 1, 1.0);

    return sprite;
}

// Warehouse creation
function renderWarehouse() {
    while (warehouseGroup.children.length > 0) {
        warehouseGroup.remove(warehouseGroup.children[0]);
    }
    coordinateLabels = [];

    if (!warehouseConfig || !warehouseConfig.length) return;

    const { length, width, height, levels, grid_size } = warehouseConfig;

    // Floor
    const floorGeometry = new THREE.PlaneGeometry(length, width);
    const floorMaterial = new THREE.MeshStandardMaterial({
        color: 0x161B22,
        side: THREE.DoubleSide,
        roughness: 0.8,
        metalness: 0.2
    });
    const floor = new THREE.Mesh(floorGeometry, floorMaterial);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    warehouseGroup.add(floor);

    // Walls
    const wallMaterial = new THREE.MeshStandardMaterial({
        color: 0x30363D,
        transparent: true,
        opacity: 0.7
    });

    // Back wall
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(length, height), wallMaterial);
    backWall.position.set(0, height/2, -width/2);
    backWall.receiveShadow = true;
    warehouseGroup.add(backWall);

    // Left wall
    const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(width, height), wallMaterial);
    leftWall.rotation.y = Math.PI / 2;
    leftWall.position.set(-length/2, height/2, 0);
    leftWall.receiveShadow = true;
    warehouseGroup.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(width, height), wallMaterial);
    rightWall.rotation.y = -Math.PI / 2;
    rightWall.position.set(length/2, height/2, 0);
    rightWall.receiveShadow = true;
    warehouseGroup.add(rightWall);

    // Grid with configurable size
    const gridMaterial = new THREE.LineBasicMaterial({ color: 0x30363D, transparent: true, opacity: 0.3 });
    const points = [];
    const halfLength = length / 2;
    const halfWidth = width / 2;

    for (let i = -halfLength; i <= halfLength; i += grid_size) {
        points.push(new THREE.Vector3(i, 0.01, -halfWidth));
        points.push(new THREE.Vector3(i, 0.01, halfWidth));
    }
    for (let i = -halfWidth; i <= halfWidth; i += grid_size) {
        points.push(new THREE.Vector3(-halfLength, 0.01, i));
        points.push(new THREE.Vector3(halfLength, 0.01, i));
    }
    const gridGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const grid = new THREE.LineSegments(gridGeometry, gridMaterial);
    warehouseGroup.add(grid);


    // Render Layers/Shelves from the specific heights
    if (warehouseConfig.layer_heights && warehouseConfig.layer_heights.length > 0) {
        const layerMaterial = new THREE.MeshStandardMaterial({
            color: 0x58A6FF,
            transparent: true,
            opacity: 0.2,
            side: THREE.DoubleSide
        });

        warehouseConfig.layer_heights.forEach(yPos => {
            if (yPos > 0) { // Don't draw a shelf on the floor
                const layerGeometry = new THREE.PlaneGeometry(length, width);
                const layerPlane = new THREE.Mesh(layerGeometry, layerMaterial);
                layerPlane.rotation.x = -Math.PI / 2;
                layerPlane.position.y = yPos;
                warehouseGroup.add(layerPlane);

                const layerGridMaterial = new THREE.LineBasicMaterial({ color: 0x58A6FF, transparent: true, opacity: 0.3 });
                const layerPoints = [];
                for (let i = -halfLength; i <= halfLength; i += grid_size) {
                    layerPoints.push(new THREE.Vector3(i, 0, -halfWidth));
                    layerPoints.push(new THREE.Vector3(i, 0, halfWidth));
                }
                for (let i = -halfWidth; i <= halfWidth; i += grid_size) {
                    layerPoints.push(new THREE.Vector3(-halfLength, 0, i));
                    layerPoints.push(new THREE.Vector3(halfLength, 0, i));
                }
                const layerGridGeometry = new THREE.BufferGeometry().setFromPoints(layerPoints);
                const layerGrid = new THREE.LineSegments(layerGridGeometry, layerGridMaterial);
                layerGrid.position.y = yPos;
                warehouseGroup.add(layerGrid);
            }
        });
    }


    // Coordinate Labels
    const labelOffset = 0.8;
    const step = Math.max(1, Math.round(length / 10));

    // X-axis labels (Length)
    for (let i = 0; i <= length; i += step) {
        const xPos = i - length / 2;
        const zPos = width / 2 + labelOffset;
        const label = createCoordinateLabel(`X: ${i.toFixed(1)}m`, new THREE.Vector3(xPos, 0.05, zPos));
        warehouseGroup.add(label);
        coordinateLabels.push(label);
    }

    // Y-axis labels (Width/Depth)
    for (let i = 0; i <= width; i += step) {
        const zPos = i - width / 2;
        const xPos = -length / 2 - labelOffset;
        const label = createCoordinateLabel(`Y: ${i.toFixed(1)}m`, new THREE.Vector3(xPos, 0.05, zPos));
        label.rotation.y = Math.PI / 2;
        warehouseGroup.add(label);
        coordinateLabels.push(label);
    }

    // Z-axis labels (Height)
    for (let i = 1; i <= height; i += step) {
        const yPos = i;
        const xPos = -length / 2 - labelOffset;
        const zPos = -width / 2;
        const label = createCoordinateLabel(`Z: ${i.toFixed(1)}m`, new THREE.Vector3(xPos, yPos, zPos));
        label.rotation.y = Math.PI / 2;
        warehouseGroup.add(label);
        coordinateLabels.push(label);
    }
}

// Exclusion zones rendering
function renderExclusionZones() {
    while (exclusionZonesGroup.children.length > 0) {
        exclusionZonesGroup.remove(exclusionZonesGroup.children[0]);
    }

    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(zones => {
            zones.forEach(zone => {
                const zoneWidth = zone.x2 - zone.x1;
                const zoneLength = zone.y2 - zone.y1;

                const geometry = new THREE.PlaneGeometry(zoneWidth, zoneLength);
                const material = new THREE.MeshBasicMaterial({
                    color: 0xff0000,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.3
                });
                const plane = new THREE.Mesh(geometry, material);
                plane.rotation.x = -Math.PI / 2;
                plane.position.set(
                    zone.x1 + zoneWidth/2 - warehouseConfig.length/2,
                    0.01,
                    zone.y1 + zoneLength/2 - warehouseConfig.width/2
                );
                exclusionZonesGroup.add(plane);

                // Add border
                const edges = new THREE.EdgesGeometry(geometry);
                const line = new THREE.LineSegments(
                    edges,
                    new THREE.LineBasicMaterial({ color: 0xff0000 })
                );
                line.rotation.x = -Math.PI / 2;
                line.position.set(
                    zone.x1 + zoneWidth/2 - warehouseConfig.length/2,
                    0.02,
                    zone.y1 + zoneLength/2 - warehouseConfig.width/2
                );
                exclusionZonesGroup.add(line);
            });
        })
        .catch(error => console.error('Error fetching exclusion zones:', error));
}

// Item visualization
function loadItemsForVisualization() {
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
        .then(res => res.json())
        .then(items => {
            // Clear and update the item data cache
            allItemsData = {};
            items.forEach(item => {
                allItemsData[item.id] = item;
            });

            // Render items from the freshly loaded data
            const solution = items.map(item => ({
                id: item.id,
                x: item.x,
                y: item.y,
                z: item.z,
                rotation: item.rotation
            }));
            renderSolution(solution);
        })
        .catch(error => console.error('Error loading items for visualization:', error));
}

// Renders items based on a given solution array.
function renderSolution(solution) {
    while (itemsGroup.children.length > 0) {
        itemsGroup.remove(itemsGroup.children[0]);
    }

    if (!solution || !warehouseConfig.length) return;

    solution.forEach(itemSolution => {
        const fullItemData = allItemsData[itemSolution.id];
        if (fullItemData) {
            // Combine the base item data with the new position/rotation from the solution
            const itemToRender = {
                ...fullItemData,
                x: itemSolution.x,
                y: itemSolution.y,
                z: itemSolution.z,
                rotation: itemSolution.rotation
            };
            createItemMesh(itemToRender);
        }
    });
}


function createItemMesh(item) {
    const { length, width, height, x, y, z, rotation, category, weight, priority, access_freq } = item;

    let color;
    const colorScheme = document.getElementById('color-scheme').value;
    switch(colorScheme) {
        case 'category':
            color = getCategoryColor(category);
            break;
        case 'weight':
            color = getWeightColor(weight);
            break;
        case 'priority':
            color = getPriorityColor(priority);
            break;
        case 'access':
            color = getAccessColor(access_freq);
            break;
        default:
            color = 0x58A6FF;
    }

    const material = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity: 0.95
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(length, height, width), material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    if (!warehouseConfig.length) return;

    // FIX: The backend now provides the CENTER coordinates (x, y).
    // The z-coordinate is for the bottom of the item, so we still add height/2.
    // The scene itself is centered at (0,0,0), so we offset by half the warehouse dimensions.
    mesh.position.set(
        x - warehouseConfig.length / 2,
        z + height / 2,
        y - warehouseConfig.width / 2
    );

    // Rotation is around the Y-axis (up).
    // The backend uses degrees, Three.js uses radians.
    mesh.rotation.y = -rotation * Math.PI / 180;

    itemsGroup.add(mesh);
}

// Color helper functions
function getCategoryColor(category) {
    const colorMap = {
        'Electronics': 0xF85149,
        'Furniture': 0x58A6FF,
        'General': 0x3FB950,
        'Heavy': 0xD29922,
        'Fragile': 0xA371F7
    };

    if (!category) return 0x666666;

    let hash = 0;
    for (let i = 0; i < category.length; i++) {
        hash = category.charCodeAt(i) + ((hash << 5) - hash);
    }

    return colorMap[category] || (hash & 0x00FFFFFF);
}

function getWeightColor(weight) {
    const intensity = Math.min(weight / 100, 1);
    return new THREE.Color(intensity, 1 - intensity, 0);
}

function getPriorityColor(priority) {
    const colors = [0xF85149, 0xD29922, 0x3FB950, 0x58A6FF, 0xA371F7];
    return colors[priority - 1] || 0x8B949E;
}

function getAccessColor(accessFreq) {
    const intensity = Math.min(accessFreq / 10, 1);
    return new THREE.Color(intensity, intensity, 1 - intensity);
}

function updateVisualization() {
    loadItemsForVisualization();
}

// UI Interaction
let activePanel = 'controls-panel';
function toggleMainPanel(panelId, button) {
    const newPanel = document.getElementById(panelId);
    const allPanels = document.querySelectorAll('.ui-panel');

    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));

    if (activePanel === panelId) {
        newPanel.classList.toggle('visible');
        activePanel = newPanel.classList.contains('visible') ? panelId : null;
    } else {
        allPanels.forEach(p => p.classList.remove('visible'));
        newPanel.classList.add('visible');
        activePanel = panelId;
    }

    if (activePanel) button.classList.add('active');

    if (panelId === 'analytics-panel' && newPanel.classList.contains('visible')) loadAnalytics();
    if (panelId === 'items-panel' && newPanel.classList.contains('visible')) loadItems();
    if (panelId === 'warehouse-panel' && newPanel.classList.contains('visible')) {
        loadWarehouseConfig();
        loadExclusionZonesList();
    }
    if (panelId === 'warehouse-manager-panel' && newPanel.classList.contains('visible')) {
        loadWarehousesList();
    }
}

function togglePanelContent(panelId) {
    document.getElementById(panelId).classList.toggle('collapsed');
}

function toggleAlgorithmParams() {
    const algo = document.getElementById('algorithm-select').value;
    document.getElementById('ga-params').style.display = (algo.includes('ga') || algo === 'compare') ? 'block' : 'none';
    document.getElementById('eo-params').style.display = (algo.includes('eo') || algo === 'compare') ? 'block' : 'none';
}

// Camera views
function viewTop() {
    gsap.to(camera.position, {duration: 0.5, x: 0, y: warehouseConfig.length, z: 0});
    gsap.to(controls.target, {duration: 0.5, x: 0, y: 0, z: 0});
}

function viewFront() {
    gsap.to(camera.position, {duration: 0.5, x: 0, y: warehouseConfig.height, z: warehouseConfig.width});
    gsap.to(controls.target, {duration: 0.5, x: 0, y: 0, z: 0});
}

function viewSide() {
    gsap.to(camera.position, {duration: 0.5, x: warehouseConfig.length, y: warehouseConfig.height, z: 0});
    gsap.to(controls.target, {duration: 0.5, x: 0, y: 0, z: 0});
}

function resetCamera() {
    controls.reset();
    camera.position.set(warehouseConfig.length * 1.5, warehouseConfig.height * 2, warehouseConfig.width * 1.5);
    controls.target.set(0, 0, 0);
    controls.update();
}

// Weight management
function updateWeightValue(element) {
    document.getElementById(`${element.id}-value`).textContent = element.value;
}

function normalizeWeights() {
    const space = document.getElementById('weight-space');
    const accessibility = document.getElementById('weight-accessibility');
    const stability = document.getElementById('weight-stability');
    const grouping = document.getElementById('weight-grouping');

    const sVal = parseFloat(space.value);
    const aVal = parseFloat(accessibility.value);
    const stVal = parseFloat(stability.value);
    const gVal = parseFloat(grouping.value);

    const total = sVal + aVal + stVal + gVal;
    if (total === 0) return;

    space.value = (sVal / total).toFixed(2);
    accessibility.value = (aVal / total).toFixed(2);
    stability.value = (stVal / total).toFixed(2);
    grouping.value = (gVal / total).toFixed(2);

    updateWeightValue(space);
    updateWeightValue(accessibility);
    updateWeightValue(stability);
    updateWeightValue(grouping);
}

// Warehouse management functions
function loadWarehouses() {
    fetch(`${API_BASE_URL}/api/warehouses`)
    .then(res => res.json())
    .then(warehousesData => {
        warehouses = warehousesData;
        updateWarehouseSelectors();
    })
    .catch(error => console.error('Error loading warehouses:', error));
}

function updateWarehouseSelectors() {
    const selectors = [
        'warehouse-select',
        'controls-warehouse-select',
        'analytics-warehouse-select',
        'items-warehouse-select',
        'data-warehouse-select',
        'config-warehouse-select',
        'copy-from-warehouse'
    ];

    selectors.forEach(selectorId => {
        const selector = document.getElementById(selectorId);
        if (selector) {
            selector.innerHTML = '';
            warehouses.forEach(warehouse => {
                const option = document.createElement('option');
                option.value = warehouse.id;
                option.textContent = warehouse.name;
                option.selected = warehouse.id === currentWarehouseId;
                selector.appendChild(option);
            });
        }
    });
}

function switchWarehouse(warehouseId) {
    currentWarehouseId = parseInt(warehouseId);

    fetch(`${API_BASE_URL}/api/warehouses/switch/${currentWarehouseId}`, {
        method: 'POST'
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification(`Switched to ${getWarehouseName(currentWarehouseId)}`, 'success');
            updateWarehouseSelectors();
            loadWarehouseConfig();
            loadItemsForVisualization();
            renderExclusionZones();
            loadAnalytics();
        }
    })
    .catch(error => showNotification('Error switching warehouse: ' + error, 'error'));
}

function getWarehouseName(id) {
    const warehouse = warehouses.find(w => w.id === id);
    return warehouse ? warehouse.name : `Warehouse ${id}`;
}

function createNewWarehouse() {
    const name = document.getElementById('new-warehouse-name').value;
    const copyFromId = document.getElementById('copy-from-warehouse').value;

    if (!name) {
        showNotification('Please enter a warehouse name', 'error');
        return;
    }

    const newWarehouse = {
        name: name,
        copy_from: copyFromId
    };

    fetch(`${API_BASE_URL}/api/warehouses`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(newWarehouse)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification('Warehouse created successfully', 'success');
            document.getElementById('new-warehouse-name').value = '';
            loadWarehouses();
            switchWarehouse(data.id);
        } else {
            showNotification('Failed to create warehouse: ' + data.error, 'error');
        }
    })
    .catch(error => showNotification('Error creating warehouse: ' + error, 'error'));
}

function deleteWarehouse(warehouseId) {
    if (warehouseId === 1) {
        showNotification('Cannot delete the default warehouse', 'error');
        return;
    }

    if (confirm(`Are you sure you want to delete "${getWarehouseName(warehouseId)}"? This will also delete all items and data associated with this warehouse.`)) {
        fetch(`${API_BASE_URL}/api/warehouses/${warehouseId}`, {
            method: 'DELETE'
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showNotification('Warehouse deleted successfully', 'success');
                loadWarehouses();
                if (currentWarehouseId === warehouseId) {
                    switchWarehouse(1);
                }
            } else {
                showNotification('Failed to delete warehouse: ' + data.error, 'error');
            }
        })
        .catch(error => showNotification('Error deleting warehouse: ' + error, 'error'));
    }
}

function loadWarehousesList() {
    const container = document.getElementById('warehouses-list');
    if (!container) return;

    container.innerHTML = '';

    if (warehouses.length === 0) {
        container.innerHTML = '<div class="item-entry">No warehouses found.</div>';
        return;
    }

    warehouses.forEach(warehouse => {
        const entry = document.createElement('div');
        entry.className = 'item-entry';
        entry.innerHTML = `
            <strong>${warehouse.name}</strong><br>
            ${warehouse.length}x${warehouse.width}x${warehouse.height}m
            <div style="display: flex; gap: 5px; margin-top: 5px;">
                <button onclick="switchWarehouse(${warehouse.id})" class="btn-small" style="background: var(--accent-primary);">Switch</button>
                ${warehouse.id !== 1 ? `<button onclick="deleteWarehouse(${warehouse.id})" class="btn-small" style="background: var(--danger);">Delete</button>` : ''}
            </div>
        `;
        container.appendChild(entry);
    });
}

// Warehouse configuration functions
function loadWarehouseConfig() {
    fetch(`${API_BASE_URL}/api/warehouse/config?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(config => {
        warehouseConfig = config;

        document.getElementById('warehouse-name').value = config.name || 'Warehouse';
        document.getElementById('warehouse-length').value = config.length;
        document.getElementById('warehouse-width').value = config.width;
        document.getElementById('warehouse-height').value = config.height;
        document.getElementById('warehouse-grid').value = config.grid_size;

        // Generate the UI for layer heights
        generateLayerHeightInputs(config.layer_heights || []);

        updateWarehouseMetrics();
        renderWarehouse();
    })
    .catch(error => console.error('Error loading warehouse config:', error));
}


function updateWarehouseConfig() {
    // Collect layer heights from the dynamically created inputs
    const layerHeightInputs = document.querySelectorAll('.layer-height-input');
    const layerHeights = Array.from(layerHeightInputs)
        .map(input => parseFloat(input.value))
        .filter(h => !isNaN(h) && h > 0);

    // Sort to maintain order
    layerHeights.sort((a, b) => a - b);

    const newConfig = {
        id: currentWarehouseId,
        name: document.getElementById('warehouse-name').value,
        length: parseFloat(document.getElementById('warehouse-length').value),
        width: parseFloat(document.getElementById('warehouse-width').value),
        height: parseFloat(document.getElementById('warehouse-height').value),
        levels: layerHeights.length + 1, // Total levels = floor + shelves
        grid_size: parseFloat(document.getElementById('warehouse-grid').value),
        layer_heights: layerHeights // Send the array of shelf heights
    };

    if (newConfig.length <= 0 || newConfig.width <= 0 || newConfig.height <= 0) {
        showNotification('Please enter valid dimensions (positive numbers)', 'error');
        return;
    }

    fetch(`${API_BASE_URL}/api/warehouse/config`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(newConfig)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification('Warehouse configuration updated successfully', 'success');
            warehouseConfig = newConfig;
            updateWarehouseMetrics();
            renderWarehouse();
            loadItemsForVisualization();
            resetCamera();
            // Refresh warehouse list to show updated name
            loadWarehouses();
        } else {
            showNotification('Failed to update warehouse configuration', 'error');
        }
    })
    .catch(error => showNotification('Error updating warehouse: ' + error, 'error'));
}

function resetWarehouseConfig() {
    const defaultConfig = {
        length: 10.0,
        width: 8.0,
        height: 5.0,
        levels: 1, // Just the floor
        grid_size: 0.5,
        layer_heights: [] // No shelves
    };

    document.getElementById('warehouse-length').value = defaultConfig.length;
    document.getElementById('warehouse-width').value = defaultConfig.width;
    document.getElementById('warehouse-height').value = defaultConfig.height;
    document.getElementById('warehouse-grid').value = defaultConfig.grid_size;

    // This will clear the inputs
    generateLayerHeightInputs(defaultConfig.layer_heights);

    updateWarehouseConfig();
}

function updateWarehouseMetrics() {
    if (!warehouseConfig) return;

    const volume = warehouseConfig.length * warehouseConfig.width * warehouseConfig.height;
    const area = warehouseConfig.length * warehouseConfig.width;

    document.getElementById('warehouse-volume').textContent = `${volume.toFixed(1)} m³`;
    document.getElementById('warehouse-area').textContent = `${area.toFixed(1)} m²`;
}

// Layer/Shelf input management
function generateLayerHeightInputs(layerHeights = null) {
    const container = document.getElementById('layer-heights-container');
    const warehouseHeight = parseFloat(document.getElementById('warehouse-height').value) || 5.0;
    let heights = [];

    if (layerHeights !== null) {
        heights = layerHeights;
        document.getElementById('warehouse-layers').value = heights.length;
    } else {
        const numLayers = parseInt(document.getElementById('warehouse-layers').value) || 0;
        const currentInputs = Array.from(container.querySelectorAll('.layer-height-input'));
        const currentValues = currentInputs.map(input => parseFloat(input.value) || 0);

        // Keep existing values, add or remove as needed
        heights = new Array(numLayers).fill(0).map((_, i) => {
            if (i < currentValues.length) return currentValues[i];
            // For new layers, distribute them reasonably
            const lastVal = i > 0 ? heights[i-1] : 0;
            return Math.min(lastVal + 1.0, warehouseHeight);
        });
    }

    container.innerHTML = ''; // Clear previous inputs

    heights.forEach((height, index) => {
        const div = document.createElement('div');
        div.className = 'form-group';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.gap = '8px';
        div.style.marginBottom = '0';

        const label = document.createElement('label');
        label.textContent = `Shelf ${index + 1} Z (m)`;
        label.style.marginBottom = '0';
        label.style.flexShrink = '0';
        label.style.width = '100px';

        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'layer-height-input';
        input.value = height.toFixed(1);
        input.min = "0.1";
        input.max = warehouseHeight.toFixed(1);
        input.step = "0.1";
        input.style.width = '100%';

        const deleteBtn = document.createElement('button');
        deleteBtn.textContent = '✕';
        deleteBtn.onclick = () => deleteLayer(index);
        deleteBtn.className = 'btn-secondary';
        deleteBtn.style.width = 'auto';
        deleteBtn.style.padding = '8px 10px';
        deleteBtn.style.lineHeight = '1';
        deleteBtn.style.background = 'var(--danger)';
        deleteBtn.setAttribute('aria-label', `Delete shelf ${index + 1}`);

        div.appendChild(label);
        div.appendChild(input);
        div.appendChild(deleteBtn);
        container.appendChild(div);
    });
}

function addLayer() {
    const numLayersInput = document.getElementById('warehouse-layers');
    numLayersInput.value = parseInt(numLayersInput.value) + 1;
    generateLayerHeightInputs();
}

function deleteLayer(index) {
    const currentInputs = Array.from(document.querySelectorAll('.layer-height-input'));
    const currentValues = currentInputs.map(input => parseFloat(input.value));

    currentValues.splice(index, 1);

    generateLayerHeightInputs(currentValues);
}

function distributeLayerHeights() {
    const warehouseHeight = parseFloat(document.getElementById('warehouse-height').value) || 5.0;
    const numLayers = parseInt(document.getElementById('warehouse-layers').value) || 0;

    if (numLayers === 0) {
        document.getElementById('layer-heights-container').innerHTML = '';
        return;
    }

    const step = warehouseHeight / (numLayers + 1);
    const heights = Array.from({length: numLayers}, (_, i) => parseFloat(((i + 1) * step).toFixed(1)));

    generateLayerHeightInputs(heights);
}

// API & Data Handling
const API_BASE_URL = 'http://127.0.0.1:5000';

function startOptimization() {
    const algorithm = document.getElementById('algorithm-select').value;
    const params = {
        warehouse_id: currentWarehouseId
    };

    if (algorithm.includes('ga') || algorithm === 'compare') {
        params.population_size = parseInt(document.getElementById('population-size').value);
        params.generations = parseInt(document.getElementById('generations').value);
    }

    if (algorithm.includes('eo') || algorithm === 'compare') {
        params.iterations = parseInt(document.getElementById('iterations').value);
    }

    params.weights = {
        space: parseFloat(document.getElementById('weight-space').value),
        accessibility: parseFloat(document.getElementById('weight-accessibility').value),
        stability: parseFloat(document.getElementById('weight-stability').value),
        grouping: parseFloat(document.getElementById('weight-grouping').value)
    };

    if (algorithm === 'compare') {
        runComparison(params);
        return;
    }

    fetch(`${API_BASE_URL}/api/optimize/${algorithm}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            document.getElementById('status-text').innerHTML = '<span class="status-indicator status-running"></span>Running';
            startProgressMonitoring();
        } else {
            showNotification('Error starting: ' + data.message, 'error');
        }
    })
    .catch(error => showNotification('API Error: ' + error, 'error'));
}

function runComparison(params) {
    const startBtn = document.getElementById('start-btn');
    startBtn.disabled = true;
    startBtn.querySelector('span').textContent = 'Comparing...';
    showNotification('Running algorithm comparison...', 'info');

    fetch(`${API_BASE_URL}/api/optimize/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            loadAnalytics();
            showNotification('Comparison complete!', 'success');
            if (!document.getElementById('analytics-panel').classList.contains('visible')) {
                toggleMainPanel('analytics-panel', document.getElementById('nav-analytics'));
            }
        } else {
            showNotification('Comparison failed: ' + data.error, 'error');
        }
    })
    .catch(error => showNotification('Comparison error: ' + error, 'error'))
    .finally(() => {
        startBtn.disabled = false;
        startBtn.querySelector('span').textContent = 'Start';
    });
}

function stopOptimization() {
    fetch(`${API_BASE_URL}/api/optimize/stop`, { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.success) showNotification('Optimization stopped.', 'info');
    });
}

function startProgressMonitoring() {
    if (optimizationInterval) clearInterval(optimizationInterval);

    let wasRunning = false;

    optimizationInterval = setInterval(() => {
        fetch(`${API_BASE_URL}/api/optimize/status`)
        .then(res => res.json())
        .then(state => {
            if (!optimizationInterval) return; // Stop if interval has been cleared

            const isRunning = state.running;
            const currentProgress = state.progress;

            if (isRunning) {
                wasRunning = true;
                if (state.start_time) {
                    const elapsed = Math.floor((Date.now() / 1000) - state.start_time);
                    lastRunningTime = `${Math.floor(elapsed/60).toString().padStart(2,'0')}:${(elapsed%60).toString().padStart(2,'0')}`;
                    document.getElementById('running-time').textContent = lastRunningTime;
                }
            }

            document.getElementById('progress-fill').style.width = currentProgress + '%';
            document.getElementById('best-fitness').textContent = state.best_fitness.toFixed(4);
            document.getElementById('current-algorithm').textContent = state.algorithm || 'None';

            // Completion condition: if it was running on a previous check and now it's not, it has finished.
            if (!isRunning && wasRunning) {
                handleOptimizationComplete();
            }

            // Real-time visualization update
            if (isRunning && state.best_solution) {
                renderSolution(state.best_solution);
            }
        })
        .catch(error => {
            console.error('Error fetching optimization status:', error);
            if (wasRunning) {
                handleOptimizationComplete();
            }
        });
    }, 1000);
}


function handleOptimizationComplete() {
    // Check if the process is already stopped to prevent multiple triggers
    if (!optimizationInterval) return;

    console.log("Handling optimization completion");

    // Clear the interval immediately
    clearInterval(optimizationInterval);
    optimizationInterval = null;

    // Force UI updates to final state
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    document.getElementById('status-text').innerHTML = '<span class="status-indicator status-stopped"></span>Stopped';
    document.getElementById('progress-fill').style.width = '100%';

    // A short delay before fetching final data to ensure DB has been written
    setTimeout(() => {
        showNotification('Optimization complete!', 'success');
        loadAnalytics();
        loadItemsForVisualization(); // Load the final, saved state from the DB
    }, 500);
}

function loadAnalytics() {
    loadCurrentMetrics();
    loadHistory();
}

function loadCurrentMetrics() {
    fetch(`${API_BASE_URL}/api/metrics/current?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(metrics => {
        const display = document.getElementById('current-metrics-display');
        if (metrics.error) return display.innerHTML = '<p>No data</p>';

        display.innerHTML = `
            <div class="stat-card">
                <div class="stat-card-title">Space Use</div>
                <div class="stat-card-value">${(metrics.space_utilization * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Accessibility</div>
                <div class="stat-card-value">${metrics.accessibility.toFixed(3)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Stability</div>
                <div class="stat-card-value">${(metrics.stability * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-title">Total Items</div>
                <div class="stat-card-value">${metrics.total_items}</div>
            </div>
        `;
    })
    .catch(e => console.error('Metrics Error:', e));
}

function loadHistory() {
    fetch(`${API_BASE_URL}/api/metrics/history?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(history => {
        const lastRunDisplay = document.getElementById('last-run-metrics');
        const tableContainer = document.getElementById('history-table-container');

        if (!history || history.length === 0) {
            lastRunDisplay.innerHTML = '<div class="stat-card"><div class="stat-card-title">Last Run</div><div class="stat-card-value">No Data</div></div>';
            tableContainer.innerHTML = '<p style="color: var(--text-dark); font-size: 0.875rem;">No optimization history found.</p>';
            if (historyChart) {
                historyChart.destroy();
            }
            return;
        }

        const lastRun = history[0];
        lastRunDisplay.innerHTML = `
            <div class="stat-card">
                <h3>Last Run: ${lastRun.algorithm}</h3>
                <div class="metric"><span>Final Fitness:</span> <span>${lastRun.fitness.toFixed(4)}</span></div>
                <div class="metric"><span>Exec. Time:</span> <span>${lastRun.execution_time.toFixed(2)}s</span></div>
            </div>
        `;

        let tableHTML = `
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Algo</th>
                        <th>Fitness</th>
                        <th>Space</th>
                        <th>Access</th>
                        <th>Stability</th>
                    </tr>
                </thead>
                <tbody>
        `;

        history.slice(0, 10).forEach(run => {
            tableHTML += `
                <tr>
                    <td>${new Date(run.timestamp).toLocaleTimeString()}</td>
                    <td>${run.algorithm}</td>
                    <td>${run.fitness.toFixed(3)}</td>
                    <td>${(run.space_utilization * 100).toFixed(1)}%</td>
                    <td>${run.accessibility.toFixed(3)}</td>
                    <td>${(run.stability * 100).toFixed(1)}%</td>
                </tr>
            `;
        });

        tableHTML += `</tbody></table>`;
        tableContainer.innerHTML = tableHTML;

        updateHistoryChart(history);

    })
    .catch(e => {
        console.error('History Error:', e);
        document.getElementById('history-table-container').innerHTML = '<p style="color: var(--danger);">Could not load history.</p>';
    });
}

function updateHistoryChart(history) {
    const latestRuns = new Map();
    history.forEach(run => {
        if (!latestRuns.has(run.algorithm)) {
            latestRuns.set(run.algorithm, run);
        }
    });

    const labels = [...latestRuns.keys()];
    const fitnessData = [];
    const spaceData = [];
    const accessibilityData = [];
    const stabilityData = [];
    const executionTimeData = [];

    labels.forEach(label => {
        const run = latestRuns.get(label);
        fitnessData.push(run.fitness.toFixed(4));
        spaceData.push((run.space_utilization * 100).toFixed(2));
        accessibilityData.push(run.accessibility.toFixed(4));
        stabilityData.push((run.stability * 100).toFixed(2));
        executionTimeData.push(run.execution_time.toFixed(2));
    });

    const chartData = {
        labels: labels,
        datasets: [
            {
                label: 'Fitness Score',
                data: fitnessData,
                backgroundColor: 'rgba(88, 166, 255, 0.7)',
                borderColor: 'rgba(88, 166, 255, 1)',
                borderWidth: 1,
                yAxisID: 'y',
            },
            {
                label: 'Space Use (%)',
                data: spaceData,
                backgroundColor: 'rgba(63, 185, 80, 0.7)',
                borderColor: 'rgba(63, 185, 80, 1)',
                borderWidth: 1,
                yAxisID: 'y1',
                hidden: true,
            },
            {
                label: 'Accessibility Score',
                data: accessibilityData,
                backgroundColor: 'rgba(210, 153, 34, 0.7)',
                borderColor: 'rgba(210, 153, 34, 1)',
                borderWidth: 1,
                yAxisID: 'y',
                hidden: true,
            },
             {
                label: 'Stability (%)',
                data: stabilityData,
                backgroundColor: 'rgba(163, 113, 247, 0.7)',
                borderColor: 'rgba(163, 113, 247, 1)',
                borderWidth: 1,
                yAxisID: 'y1',
                hidden: true,
            },
            {
                label: 'Exec. Time (s)',
                data: executionTimeData,
                backgroundColor: 'rgba(248, 81, 73, 0.7)',
                borderColor: 'rgba(248, 81, 73, 1)',
                borderWidth: 1,
                yAxisID: 'y',
                hidden: true,
            }
        ]
    };

    const ctx = document.getElementById('historyChart').getContext('2d');

    if (historyChart) {
        historyChart.destroy();
    }

    historyChart = new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Latest Algorithm Performance Comparison',
                    color: '#C9D1D9',
                    font: { size: 16 }
                },
                legend: {
                    labels: { color: '#C9D1D9' }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: { color: '#8B949E' },
                    grid: { color: 'rgba(48, 54, 61, 0.5)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Score / Time (s)',
                        color: '#C9D1D9'
                    },
                    ticks: { color: '#8B949E' },
                    grid: { color: 'rgba(48, 54, 61, 0.5)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percentage (%)',
                        color: '#C9D1D9'
                    },
                    ticks: { color: '#8B949E' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

// Data management functions
function uploadCSV() {
    const fileInput = document.getElementById('csv-upload');
    if (!fileInput.files.length) return showNotification('Please select a CSV file', 'error');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch(`${API_BASE_URL}/api/upload-csv?warehouse_id=${currentWarehouseId}`, { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification('Upload successful', 'success');
            fileInput.value = '';
            loadItems();
            loadItemsForVisualization();
        } else {
            showNotification('Upload failed: ' + data.error, 'error');
        }
    })
    .catch(error => showNotification('Upload error: ' + error, 'error'));
}

function exportCSV() {
    window.open(`${API_BASE_URL}/api/export-csv?warehouse_id=${currentWarehouseId}`, '_blank');
}

function loadSampleData() {
    fetch(`${API_BASE_URL}/api/load-sample-data?warehouse_id=${currentWarehouseId}`, { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification('Sample data loaded', 'success');
            loadItems();
            loadItemsForVisualization();
        } else {
            showNotification('Failed to load sample data', 'error');
        }
    })
    .catch(error => showNotification('Error loading sample data: ' + error, 'error'));
}

function clearData() {
    if (confirm('Are you sure you want to clear all item data?')) {
        fetch(`${API_BASE_URL}/api/clear-data?warehouse_id=${currentWarehouseId}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showNotification('Data cleared', 'success');
                loadItems();
                loadItemsForVisualization();
            } else {
                showNotification('Failed to clear data', 'error');
            }
        })
        .catch(error => showNotification('Error clearing data: ' + error, 'error'));
    }
}

function loadItems() {
    fetch(`${API_BASE_URL}/api/items?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(items => {
        const list = document.getElementById('items-list');
        list.innerHTML = items.length > 0 ? '' : '<div class="item-entry">No items loaded.</div>';

        items.forEach(item => {
            const entry = document.createElement('div');
            entry.className = 'item-entry';
            entry.innerHTML = `
                <strong>${item.id}</strong> - ${item.category || 'N/A'}<br>
                ${item.length}x${item.width}x${item.height}m | ${item.weight}kg
            `;
            list.appendChild(entry);
        });
    })
    .catch(error => console.error('Error loading items:', error));
}

// Exclusion zone management
function addExclusionZone() {
    const name = document.getElementById('zone-name').value;
    if (!name) {
        showNotification('Please enter a zone name', 'error');
        return;
    }

    const x1 = parseFloat(document.getElementById('zone-x1').value);
    const y1 = parseFloat(document.getElementById('zone-y1').value);
    const x2 = parseFloat(document.getElementById('zone-x2').value);
    const y2 = parseFloat(document.getElementById('zone-y2').value);

    if (isNaN(x1) || isNaN(y1) || isNaN(x2) || isNaN(y2)) {
        showNotification('Invalid coordinates. Please enter numbers.', 'error');
        return;
    }

    const zone = {
        name,
        x1: Math.min(x1, x2),
        y1: Math.min(y1, y2),
        x2: Math.max(x1, x2),
        y2: Math.max(y1, y2),
        zone_type: 'exclusion',
        warehouse_id: currentWarehouseId
    };

    fetch(`${API_BASE_URL}/api/warehouse/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zone)
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            showNotification('Exclusion zone added', 'success');
            renderExclusionZones();
            loadExclusionZonesList();
            // Clear input fields
            document.getElementById('zone-name').value = '';
            document.getElementById('zone-x1').value = '';
            document.getElementById('zone-y1').value = '';
            document.getElementById('zone-x2').value = '';
            document.getElementById('zone-y2').value = '';
        } else {
            showNotification('Failed to add zone: ' + data.error, 'error');
        }
    })
    .catch(error => showNotification('Error: ' + error, 'error'));
}

function deleteExclusionZone(zoneId) {
    if (confirm('Are you sure you want to delete this exclusion zone?')) {
        fetch(`${API_BASE_URL}/api/warehouse/zones/${zoneId}?warehouse_id=${currentWarehouseId}`, {
            method: 'DELETE'
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showNotification('Exclusion zone deleted', 'success');
                renderExclusionZones();
                loadExclusionZonesList();
            } else {
                showNotification('Failed to delete zone', 'error');
            }
        })
    }
}

function loadExclusionZonesList() {
    fetch(`${API_BASE_URL}/api/warehouse/zones?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(zones => {
        const container = document.getElementById('exclusion-zones-list');
        if (!container) return;

        container.innerHTML = '';

        if (zones.length === 0) {
            container.innerHTML = '<div class="item-entry">No exclusion zones defined.</div>';
            return;
        }

        zones.forEach(zone => {
            const entry = document.createElement('div');
            entry.className = 'item-entry';
            entry.innerHTML = `
                <strong>${zone.name}</strong><br>
                Area: (${zone.x1}, ${zone.y1}) to (${zone.x2}, ${zone.y2})
                <button onclick="deleteExclusionZone(${zone.id})"
                        style="background:var(--danger); margin-top:5px; padding:4px 8px; font-size:0.8rem; width: auto; float: right;">
                    Delete
                </button>
            `;
            container.appendChild(entry);
        });
    })
    .catch(error => console.error('Error loading exclusion zones:', error));
}

function showNotification(message, type = 'info') {
    const el = document.getElementById('notification');
    el.textContent = message;
    el.className = type;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 3000);
}

// App Initialization
document.addEventListener('DOMContentLoaded', function() {
    initThreeJS();
    toggleAlgorithmParams();

    // Load warehouses first, then load the current warehouse config
    loadWarehouses();

    fetch(`${API_BASE_URL}/api/warehouse/config?warehouse_id=${currentWarehouseId}`)
    .then(res => res.json())
    .then(config => {
        warehouseConfig = config;

        if (document.getElementById('warehouse-length')) {
            loadWarehouseConfig();
        }

        renderWarehouse();
        loadAnalytics();
        loadItemsForVisualization();
        renderExclusionZones();

        controls.target.set(0, warehouseConfig.height / 2, 0);
        camera.position.set(
            warehouseConfig.length * 1.2,
            warehouseConfig.height * 1.5,
            warehouseConfig.width * 1.2
        );
        controls.update();
    })
    .catch(error => {
        console.error("Connection Error:", error);
        showNotification("Could not connect to backend.", "error");
    });

    startProgressMonitoring();
});