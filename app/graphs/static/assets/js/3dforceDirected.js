
function forceDGraph(dataset) {

          <script src="https://unpkg.com/d3-dsv"></script>
  <script src="https://unpkg.com/dat.gui"></script>
  <script src="https://unpkg.com/d3-octree"></script>
  <script src="https://unpkg.com/d3-force-3d"></script>
  <script src="https://unpkg.com/three-spritetext"></script>
  <script src="https://unpkg.com/3d-force-graph"></script>
  <script src="https://unpkg.com/three"></script>
  <script src="https://unpkg.com/three-spritetext"></script>

    import {CSS2DRenderer, CSS2DObject} from 'https://unpkg.com/three/examples/jsm/renderers/CSS2DRenderer.js';

    // controls
    //const controls = { 'DAG Orientation': 'td'};
    //const gui = new dat.GUI();
    //gui.add(controls, 'DAG Orientation', ['td', 'bu', 'lr', 'rl', 'zout', 'zin', 'radialout', 'radialin', null])
    // .onChange(orientation => graph && graph.dagMode(orientation));

    // graph config
    const NODE_REL_SIZE = 1;
    const elem = document.getElementById("graph");
    const Graph = ForceGraph3D({extraRenderers: [new CSS2DRenderer()]})(elem)
        .dagMode('td')
        .dagLevelDistance(20)
        .backgroundColor('#101020')
        .linkColor(() => 'rgba(255,255,255,0.2)')
        .nodeAutoColorBy('group')
        .nodeRelSize(NODE_REL_SIZE)
        .nodeId('path')
        .nodeVal('size')
        //.style("font-size", 12)
        .nodeLabel('country')
        .nodeAutoColorBy('country')
        .nodeOpacity(1.0)
        //.nodelabel
        .linkDirectionalParticles(2)
        .linkDirectionalParticleWidth(0.8)
        .linkDirectionalParticleSpeed(0.01)
        .d3Force('collision', d3.forceCollide(node => Math.cbrt(node.size) * NODE_REL_SIZE))
        .d3VelocityDecay(0.5)
        .onNodeClick(node => {
            // Aim at node from outside it
            const distance = 40;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

            const newPos = node.x || node.y || node.z
                ? {x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio}
                : {x: 0, y: 0, z: distance}; // special case if node is in (0,0,0)

            Graph.cameraPosition(
                newPos, // new position
                node, // lookAt ({ x, y, z })
                3000  // ms transition duration
            );
        })
        .nodeThreeObject(node => {
            const nodeEl = document.createElement('div');
            //nodeEl.textContent = node.dateTo;
            nodeEl.style.color = node.color;
            nodeEl.style.fontSize = "35px";
            //nodeEl.className = 'node-label';
            return new CSS2DObject(nodeEl);
        })
        .nodeThreeObjectExtend(true)
        //.nodeThreeObject(node => {
        //const sprite = new SpriteText(node.country);
        //sprite.material.depthWrite = false;
        //sprite.color = 'lightsteelblue';
        //sprite.textHeight = 8;
        //return sprite;
        //})
        .onNodeDragEnd(node => {
            node.fx = node.x;
            node.fy = node.y;
            node.fz = node.z;
        });

    // Decrease repel intensity
    Graph.d3Force('charge').strength(-25);

    function waitforme(millisec) {
        return new Promise(resolve => {
            setTimeout(() => {
                resolve('')
            }, millisec);
        })
    }

    fetch(dataset)
        .then(r => r.text())
        .then(d3.csvParse)
        .then(data => {
            const nodes = [], links = [];
            let i = 0;
            data.forEach(({
                              size, path, dateFrom, group, country, culture, economic, language,
                              source, dateTo, url, polalign, i
                          }, index) => {
                setTimeout(() => {
                    console.log("path is here")
                    console.log(path)
                    const levels = path.split('/'), level = levels.length - 1, module = level > 0 ? levels[1] : null,
                        leaf = levels.pop(), parent = levels.join('/');
                    console.log("levels are here")
                    console.log(levels)
                    const node = {
                        polalign,
                        url,
                        dateTo,
                        source,
                        language,
                        country,
                        culture,
                        economic,
                        group,
                        dateFrom,
                        path,
                        leaf,
                        module,
                        size: +size || 80,
                        level
                    };

                    nodes.push(node);

                    if (parent) {
                        if (links.length == 0) {
                            //console.log(links)
                            //console.log(path)
                            //console.log(parent)
                        }
                        links.push({source: parent, target: path, targetNode: node});
                        Graph.graphData({
                            nodes: [...nodes, {parent}],
                            links: [...links, {source: parent, target: path, targetNode: node}]
                        });
                        console.log("Inside")
                        i = i + 1
                    }
                }, index * 1000);
            });
        });
}