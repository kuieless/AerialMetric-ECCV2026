window.HELP_IMPROVE_VIDEOJS = false;

const METRIC_COMPARE_MOTION = {
  start: 25,
  min: 25,
  max: 75,
  step: 0.20,
  framesPerTurn: 300,
};

function initAerialTeaser() {
  const hero = document.querySelector('.metric-hero');
  const showcase = document.querySelector('.metric-stage-shell');
  if (!showcase) {
    return;
  }

  const stage = showcase.querySelector('.metric-stage');
  const compare = showcase.querySelector('.metric-compare-viewport');
  const rgbImage = showcase.querySelector('[data-role="rgb"]');
  const gtImage = showcase.querySelector('[data-role="gt"]');
  const baselineImage = showcase.querySelector('[data-role="baseline"]');
  const oursImage = showcase.querySelector('[data-role="ours"]');
  const sceneButtons = document.querySelectorAll('.metric-example-rail .example-card');

  const scenes = {
    teaser1: {
      label: 'Example 01',
      rgb: './static/images/teaser1/Input_RGB.png',
      gt: './static/images/teaser1/Ground_Truth.png',
      baseline: './static/images/teaser1/Baseline.png',
      ours: './static/images/teaser1/LoRA-96.png',
    },
    teaser2: {
      label: 'Example 02',
      rgb: './static/images/teaser2/Input_RGB.png',
      gt: './static/images/teaser2/Ground_Truth.png',
      baseline: './static/images/teaser2/Baseline.png',
      ours: './static/images/teaser2/LoRA-96.png',
    },
  };

  let currentScene = 'teaser1';
  let splitBase = METRIC_COMPARE_MOTION.start;
  let splitDir = 1;
  let splitFrame = 0;

  function updateTeaser() {
    hero.style.setProperty('--metric-backdrop', `url('${scenes[currentScene].ours}')`);
    stage.dataset.currentScene = currentScene;
    stage.classList.add('is-switching');
    window.setTimeout(() => {
      const scene = scenes[currentScene];
      rgbImage.src = scene.rgb;
      rgbImage.alt = `${scene.label} aerial RGB input`;
      gtImage.src = scene.gt;
      gtImage.alt = `${scene.label} ground truth depth`;
      baselineImage.src = scene.baseline;
      baselineImage.alt = `${scene.label} MoGe2 baseline depth result`;
      oursImage.src = scene.ours;
      oursImage.alt = `${scene.label} MoGe2-Aerial depth result`;
      sceneButtons.forEach((item) => {
        item.classList.toggle('is-active', item.dataset.scene === currentScene);
      });
      window.requestAnimationFrame(() => {
        stage.classList.remove('is-switching');
      });
    }, 180);
  }

  function tickSplit() {
    splitFrame += 1;
    if (splitFrame % METRIC_COMPARE_MOTION.framesPerTurn === 0) {
      splitDir *= -1;
    }
    splitBase += splitDir * METRIC_COMPARE_MOTION.step;
    const clamped = Math.max(METRIC_COMPARE_MOTION.min, Math.min(METRIC_COMPARE_MOTION.max, splitBase));
    compare.style.setProperty('--split', `${clamped}%`);
    window.requestAnimationFrame(tickSplit);
  }

  sceneButtons.forEach((button) => {
    button.addEventListener('click', () => {
      currentScene = button.dataset.scene;
      updateTeaser();
    });
  });

  updateTeaser();
  window.requestAnimationFrame(tickSplit);
}

function initDepthExplorer() {
  const root = document.querySelector('.depth-explorer');
  const demoData = window.AERIAL_DEPTH_DEMO;
  if (!root || !demoData || !demoData.scenes || !window.THREE || !window.THREE.OrbitControls) {
    return;
  }

  const imageEl = root.querySelector('.depth-explorer-image');
  const markersEl = root.querySelector('[data-role="markers"]');
  const activeLabel = root.querySelector('[data-role="active-label"]');
  const valueGt = root.querySelector('[data-role="value-gt"]');
  const valueBaseline = root.querySelector('[data-role="value-baseline"]');
  const valueLora = root.querySelector('[data-role="value-lora96"]');
  const cardGt = root.querySelector('[data-role="card-gt"]');
  const cardBaseline = root.querySelector('[data-role="card-baseline"]');
  const cardLora = root.querySelector('[data-role="card-lora96"]');
  const sceneButtons = root.querySelectorAll('[data-scene]');
  const views = {
    gt: {
      canvas: root.querySelector('[data-role="canvas-gt"]'),
      bubble: root.querySelector('[data-role="bubble-gt"]'),
      stage: root.querySelector('[data-role="canvas-gt"]').closest('.depth-explorer-canvas-stage'),
    },
    baseline: {
      canvas: root.querySelector('[data-role="canvas-baseline"]'),
      bubble: root.querySelector('[data-role="bubble-baseline"]'),
      stage: root.querySelector('[data-role="canvas-baseline"]').closest('.depth-explorer-canvas-stage'),
    },
    lora96: {
      canvas: root.querySelector('[data-role="canvas-lora96"]'),
      bubble: root.querySelector('[data-role="bubble-lora96"]'),
      stage: root.querySelector('[data-role="canvas-lora96"]').closest('.depth-explorer-canvas-stage'),
    },
  };
  const THREE = window.THREE;
  Object.values(views).forEach((view) => {
    const dot = document.createElement('div');
    dot.className = 'depth-explorer-probe-dot';
    view.stage.appendChild(dot);
    view.dot = dot;
  });

  const state = {
    currentScene: 'scene1',
    activePoint: 0,
    activeProbe: null,
    activeProbeWorlds: {},
    activeMarkerIndex: 0,
    clouds: {},
    pointButtons: [],
    pointWorlds: {},
    syncLock: false,
    currentData: null,
  };

  function parseCloud(flat, modelKey) {
    const points = [];
    for (let i = 0; i < flat.length; i += 6) {
      points.push({
        x: flat[i],
        y: flat[i + 1],
        z: flat[i + 2],
        r: flat[i + 3],
        g: flat[i + 4],
        b: flat[i + 5],
        size: modelKey === 'lora96' ? 2.05 : 1.8,
      });
    }
    return points;
  }

  function pointToWorld(pointDef, modelKey) {
    const current = state.currentData;
    const depth = pointDef.depths[modelKey];
    const { fx, fy, cx, cy } = current.intrinsics;
    const px = pointDef.x * (current.width - 1);
    const py = pointDef.y * (current.height - 1);
    return {
      x: ((px - cx) / fx) * depth,
      y: -((py - cy) / fy) * depth,
      z: depth,
      depth,
    };
  }

  function computeSceneTransform() {
    const gt = state.clouds.gt;
    if (!gt || !gt.length) {
      return { center: { x: 0, y: 0, z: 0 }, scale: 1 };
    }
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < gt.length; i += 1) {
      const p = gt[i];
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.z < minZ) minZ = p.z;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
      if (p.z > maxZ) maxZ = p.z;
    }
    const center = {
      x: (minX + maxX) * 0.5,
      y: (minY + maxY) * 0.5,
      z: (minZ + maxZ) * 0.5,
    };
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1);
    return {
      center,
      scale: 2.6 / span,
    };
  }

  function normalizePoint(point, transform) {
    return {
      x: -(point.x - transform.center.x) * transform.scale,
      y: (point.y - transform.center.y) * transform.scale,
      z: (point.z - transform.center.z) * transform.scale,
      r: point.r,
      g: point.g,
      b: point.b,
    };
  }

  function resizeViews() {
    Object.values(views).forEach((view) => {
      const rect = view.stage.getBoundingClientRect();
      view.camera.aspect = rect.width / rect.height;
      view.camera.updateProjectionMatrix();
      view.renderer.setSize(rect.width, rect.height, false);
    });
  }

  function updateReadout() {
    const point = state.activeProbe;
    const baselineDelta = Math.abs(point.depths.baseline - point.depths.gt);
    const loraDelta = Math.abs(point.depths.lora96 - point.depths.gt);
    activeLabel.textContent = `${point.id} · ${point.label}`;
    valueGt.textContent = `${point.depths.gt.toFixed(1)} m`;
    valueBaseline.textContent = `${point.depths.baseline.toFixed(1)} m`;
    valueLora.textContent = `${point.depths.lora96.toFixed(1)} m`;
    cardGt.classList.add('is-reference');
    cardBaseline.classList.toggle('is-closest', baselineDelta < loraDelta);
    cardBaseline.classList.toggle('is-far', baselineDelta >= loraDelta);
    cardLora.classList.toggle('is-closest', loraDelta <= baselineDelta);
    cardLora.classList.toggle('is-far', loraDelta > baselineDelta);
    state.pointButtons.forEach((button, index) => {
      button.classList.toggle('is-active', index === state.activeMarkerIndex);
    });
  }

  function updateBubbles() {
    const point = state.activeProbe;
    Object.entries(views).forEach(([modelKey, view]) => {
      const world = state.activeProbeWorlds[modelKey];
      const projected = world.clone().project(view.camera);
      const rect = view.stage.getBoundingClientRect();
      const x = (projected.x * 0.5 + 0.5) * rect.width;
      const y = (-projected.y * 0.5 + 0.5) * rect.height;
      view.bubble.style.opacity = projected.z < 1 ? '1' : '0';
      view.bubble.style.left = `${x}px`;
      view.bubble.style.top = `${y}px`;
      view.bubble.textContent = `${point.id} · ${point.depths[modelKey].toFixed(1)} m`;
      view.dot.style.opacity = projected.z < 1 ? '1' : '0';
      view.dot.style.left = `${x}px`;
      view.dot.style.top = `${y}px`;
    });
  }

  function clearScene() {
    markersEl.innerHTML = '';
    state.pointButtons = [];
    Object.values(views).forEach((view) => {
      if (view.pointCloud) {
        view.scene.remove(view.pointCloud);
        view.pointCloud.geometry.dispose();
        view.pointCloud.material.dispose();
        view.pointCloud = null;
      }
    });
  }

  function syncCamera(sourceKey) {
    if (state.syncLock) {
      return;
    }
    state.syncLock = true;
    const source = views[sourceKey];
    const srcPos = source.camera.position.clone();
    const srcTarget = source.controls.target.clone();
    Object.entries(views).forEach(([key, view]) => {
      if (key === sourceKey) {
        return;
      }
      view.camera.position.copy(srcPos);
      view.controls.target.copy(srcTarget);
      view.controls.update();
    });
    state.syncLock = false;
  }

  function animate() {
    Object.values(views).forEach((view) => {
      view.controls.update();
      view.renderer.render(view.scene, view.camera);
    });
    updateBubbles();
    window.requestAnimationFrame(animate);
  }

  function createPointCloud(modelKey, transform) {
    const normalized = state.clouds[modelKey].map((point) => normalizePoint(point, transform));
    const positions = new Float32Array(normalized.length * 3);
    const colors = new Float32Array(normalized.length * 3);
    for (let i = 0; i < normalized.length; i += 1) {
      const p = normalized[i];
      const offset = i * 3;
      positions[offset] = p.x;
      positions[offset + 1] = p.y;
      positions[offset + 2] = p.z;
      colors[offset] = p.r / 255;
      colors[offset + 1] = p.g / 255;
      colors[offset + 2] = p.b / 255;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.PointsMaterial({
      size: modelKey === 'lora96' ? 0.03 : 0.026,
      vertexColors: true,
      sizeAttenuation: true,
    });
    return new THREE.Points(geometry, material);
  }

  function buildViewScaffold(modelKey) {
    const view = views[modelKey];
    view.scene = new THREE.Scene();
    view.scene.background = new THREE.Color(0xf8fbff);

    view.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    view.camera.position.set(1.8, 1.45, 3.5);

    view.renderer = new THREE.WebGLRenderer({
      canvas: view.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    view.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const ambient = new THREE.AmbientLight(0xffffff, 1.2);
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.65);
    keyLight.position.set(2, 3, 4);
    view.scene.add(ambient, keyLight);

    view.controls = new THREE.OrbitControls(view.camera, view.renderer.domElement);
    view.controls.enablePan = false;
    view.controls.enableDamping = true;
    view.controls.dampingFactor = 0.08;
    view.controls.minDistance = 1.6;
    view.controls.maxDistance = 7.5;
    view.controls.target.set(0, 0, 0);
    view.controls.update();
    view.controls.addEventListener('change', () => syncCamera(modelKey));
  }

  function buildMarkers() {
    state.currentData.points.forEach((point, index) => {
      const button = document.createElement('button');
      button.className = 'depth-explorer-marker';
      button.type = 'button';
      button.style.left = `${point.x * 100}%`;
      button.style.top = `${point.y * 100}%`;
      button.innerHTML = `
        <span class="depth-explorer-marker-dot"></span>
        <span class="depth-explorer-marker-label">${point.id}</span>
      `;
      button.addEventListener('click', (event) => {
        event.stopPropagation();
        setActiveProbe(state.currentData.points[index], index);
      });
      markersEl.appendChild(button);
      state.pointButtons.push(button);
    });
  }

  function computeProbeWorlds(pointDef) {
    const worlds = {};
    Object.keys(views).forEach((modelKey) => {
      const world = pointToWorld(pointDef, modelKey);
      const normalized = normalizePoint(world, state.sceneTransform);
      worlds[modelKey] = new THREE.Vector3(normalized.x, normalized.y, normalized.z);
    });
    return worlds;
  }

  function setActiveProbe(pointDef, markerIndex) {
    state.activeProbe = pointDef;
    state.activeMarkerIndex = markerIndex;
    state.activeProbeWorlds = computeProbeWorlds(pointDef);
    updateReadout();
  }

  function loadScene(sceneKey) {
    state.currentScene = sceneKey;
    state.currentData = demoData.scenes[sceneKey];
    state.activePoint = 0;
    imageEl.src = state.currentData.image;
    imageEl.alt = `${state.currentData.label} RGB scene for the 3D depth demo`;
    sceneButtons.forEach((button) => {
      button.classList.toggle('is-active', button.dataset.scene === sceneKey);
    });
    clearScene();
    state.clouds.gt = parseCloud(state.currentData.clouds.gt, 'gt');
    state.clouds.baseline = parseCloud(state.currentData.clouds.baseline, 'baseline');
    state.clouds.lora96 = parseCloud(state.currentData.clouds.lora96, 'lora96');
    const transform = computeSceneTransform();
    state.sceneTransform = transform;
    Object.keys(views).forEach((modelKey) => {
      const view = views[modelKey];
      view.pointCloud = createPointCloud(modelKey, transform);
      view.scene.add(view.pointCloud);
      state.pointWorlds[modelKey] = state.currentData.points.map((pointDef) => {
        const world = pointToWorld(pointDef, modelKey);
        const normalized = normalizePoint(world, transform);
        return new THREE.Vector3(normalized.x, normalized.y, normalized.z);
      });
    });
    buildMarkers();
    setActiveProbe(state.currentData.points[0], 0);
  }

  sceneButtons.forEach((button) => {
    button.addEventListener('click', () => {
      loadScene(button.dataset.scene);
    });
  });

  buildViewScaffold('gt');
  buildViewScaffold('baseline');
  buildViewScaffold('lora96');
  loadScene(state.currentScene);
  window.addEventListener('resize', resizeViews);
  resizeViews();
  animate();
}

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    initAerialTeaser();
    initDepthExplorer();

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
