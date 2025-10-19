// QSDT理论网站 - Three.js 3D动画系统

class QSDTAnimationSystem {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.animationId = null;
        this.currentAnimation = null;
        this.parameters = {};
        
        this.init();
    }
    
    init() {
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.setupControls();
        this.animate();
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        
        // 添加雾效果
        this.scene.fog = new THREE.Fog(0x0a0a0a, 50, 200);
    }
    
    setupCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 10);
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
        
        // 响应式处理
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupLighting() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
        this.scene.add(ambientLight);
        
        // 主光源
        const directionalLight = new THREE.DirectionalLight(0x0066ff, 1);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // 点光源
        const pointLight = new THREE.PointLight(0xff6600, 0.5, 100);
        pointLight.position.set(-10, -10, 10);
        this.scene.add(pointLight);
    }
    
    setupControls() {
        // 轨道控制器
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.enablePan = true;
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        if (this.currentAnimation && this.currentAnimation.update) {
            this.currentAnimation.update();
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    // 量子空间网络动画
    createQuantumNetworkAnimation() {
        this.clearScene();
        
        const network = {
            nodes: [],
            connections: [],
            update: () => this.updateQuantumNetwork()
        };
        
        // 创建量子空间节点
        const nodeGeometry = new THREE.SphereGeometry(0.1, 16, 16);
        const nodeMaterial = new THREE.MeshPhongMaterial({ 
            color: 0x0066ff,
            emissive: 0x001133,
            transparent: true,
            opacity: 0.8
        });
        
        // 创建网络节点
        for (let i = 0; i < 50; i++) {
            const node = new THREE.Mesh(nodeGeometry, nodeMaterial.clone());
            node.position.set(
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 20
            );
            node.userData = {
                originalPosition: node.position.clone(),
                phase: Math.random() * Math.PI * 2,
                speed: 0.01 + Math.random() * 0.02
            };
            this.scene.add(node);
            network.nodes.push(node);
        }
        
        // 创建连接线
        const lineGeometry = new THREE.BufferGeometry();
        const lineMaterial = new THREE.LineBasicMaterial({ 
            color: 0x3388ff,
            transparent: true,
            opacity: 0.3
        });
        
        for (let i = 0; i < network.nodes.length; i++) {
            for (let j = i + 1; j < network.nodes.length; j++) {
                const distance = network.nodes[i].position.distanceTo(network.nodes[j].position);
                if (distance < 5) {
                    const line = new THREE.Line(
                        new THREE.BufferGeometry().setFromPoints([
                            network.nodes[i].position,
                            network.nodes[j].position
                        ]),
                        lineMaterial
                    );
                    this.scene.add(line);
                    network.connections.push(line);
                }
            }
        }
        
        this.currentAnimation = network;
        return network;
    }
    
    updateQuantumNetwork() {
        if (!this.currentAnimation) return;
        
        const time = Date.now() * 0.001;
        
        // 更新节点位置
        this.currentAnimation.nodes.forEach((node, index) => {
            const userData = node.userData;
            const offset = new THREE.Vector3(
                Math.sin(time * userData.speed + userData.phase) * 0.5,
                Math.cos(time * userData.speed + userData.phase) * 0.3,
                Math.sin(time * userData.speed * 1.5 + userData.phase) * 0.2
            );
            
            node.position.copy(userData.originalPosition).add(offset);
            
            // 根据距离调整透明度
            const distance = node.position.distanceTo(this.camera.position);
            node.material.opacity = Math.max(0.3, 1 - distance / 30);
        });
        
        // 更新连接线
        this.currentAnimation.connections.forEach(connection => {
            const positions = connection.geometry.attributes.position.array;
            const node1 = this.currentAnimation.nodes[0];
            const node2 = this.currentAnimation.nodes[1];
            
            if (node1 && node2) {
                positions[0] = node1.position.x;
                positions[1] = node1.position.y;
                positions[2] = node1.position.z;
                positions[3] = node2.position.x;
                positions[4] = node2.position.y;
                positions[5] = node2.position.z;
                connection.geometry.attributes.position.needsUpdate = true;
            }
        });
    }
    
    // 对称性破缺动画
    createSymmetryBreakingAnimation() {
        this.clearScene();
        
        const breaking = {
            particles: [],
            phase: 0,
            update: () => this.updateSymmetryBreaking()
        };
        
        // 创建对称粒子系统
        const particleGeometry = new THREE.SphereGeometry(0.05, 8, 8);
        const particleMaterial = new THREE.MeshPhongMaterial({ 
            color: 0xff6600,
            emissive: 0x331100
        });
        
        // 初始对称状态
        for (let i = 0; i < 100; i++) {
            const particle = new THREE.Mesh(particleGeometry, particleMaterial.clone());
            const angle = (i / 100) * Math.PI * 2;
            const radius = 3 + Math.random() * 2;
            particle.position.set(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                (Math.random() - 0.5) * 4
            );
            particle.userData = {
                originalPosition: particle.position.clone(),
                group: Math.floor(i / 25) // 分为4组
            };
            this.scene.add(particle);
            breaking.particles.push(particle);
        }
        
        this.currentAnimation = breaking;
        return breaking;
    }
    
    updateSymmetryBreaking() {
        if (!this.currentAnimation) return;
        
        const time = Date.now() * 0.001;
        this.currentAnimation.phase = Math.min(time * 0.5, 1);
        
        this.currentAnimation.particles.forEach((particle, index) => {
            const userData = particle.userData;
            const group = userData.group;
            const phase = this.currentAnimation.phase;
            
            // 根据组别和相位计算新位置
            let newPosition = userData.originalPosition.clone();
            
            if (phase > 0.2) {
                // 开始破缺
                const breakOffset = new THREE.Vector3(
                    Math.sin(time + group) * phase * 2,
                    Math.cos(time + group) * phase * 1.5,
                    Math.sin(time * 2 + group) * phase
                );
                newPosition.add(breakOffset);
                
                // 改变颜色表示破缺
                const breakColor = new THREE.Color().lerpColors(
                    new THREE.Color(0xff6600),
                    new THREE.Color(0x0066ff),
                    phase
                );
                particle.material.color.copy(breakColor);
            }
            
            particle.position.copy(newPosition);
        });
    }
    
    // 粒子涌现动画
    createParticleEmergenceAnimation() {
        this.clearScene();
        
        const emergence = {
            particles: [],
            time: 0,
            update: () => this.updateParticleEmergence()
        };
        
        // 创建能量场
        const fieldGeometry = new THREE.PlaneGeometry(20, 20);
        const fieldMaterial = new THREE.MeshBasicMaterial({
            color: 0x001133,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        const field = new THREE.Mesh(fieldGeometry, fieldMaterial);
        field.rotation.x = -Math.PI / 2;
        this.scene.add(field);
        
        this.currentAnimation = emergence;
        return emergence;
    }
    
    updateParticleEmergence() {
        if (!this.currentAnimation) return;
        
        const time = Date.now() * 0.001;
        this.currentAnimation.time = time;
        
        // 随机生成新粒子
        if (Math.random() < 0.1) {
            const particleGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const particleMaterial = new THREE.MeshPhongMaterial({
                color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
                emissive: new THREE.Color().setHSL(Math.random(), 0.8, 0.2)
            });
            
            const particle = new THREE.Mesh(particleGeometry, particleMaterial);
            particle.position.set(
                (Math.random() - 0.5) * 15,
                (Math.random() - 0.5) * 15,
                (Math.random() - 0.5) * 15
            );
            particle.userData = {
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.02,
                    (Math.random() - 0.5) * 0.02,
                    (Math.random() - 0.5) * 0.02
                ),
                life: 1.0
            };
            
            this.scene.add(particle);
            this.currentAnimation.particles.push(particle);
        }
        
        // 更新现有粒子
        this.currentAnimation.particles.forEach((particle, index) => {
            const userData = particle.userData;
            userData.life -= 0.005;
            
            if (userData.life <= 0) {
                this.scene.remove(particle);
                this.currentAnimation.particles.splice(index, 1);
                return;
            }
            
            particle.position.add(userData.velocity);
            particle.material.opacity = userData.life;
            particle.scale.setScalar(userData.life);
        });
    }
    
    // 时间二元性动画
    createTimeDualityAnimation() {
        this.clearScene();
        
        const duality = {
            objectiveTime: [],
            subjectiveTime: [],
            time: 0,
            update: () => this.updateTimeDuality()
        };
        
        // 客观时间 - 网络状态演化
        const objectiveGeometry = new THREE.BoxGeometry(0.2, 0.2, 0.2);
        const objectiveMaterial = new THREE.MeshPhongMaterial({ color: 0x0066ff });
        
        for (let i = 0; i < 20; i++) {
            const box = new THREE.Mesh(objectiveGeometry, objectiveMaterial.clone());
            box.position.set(i * 0.5 - 5, 0, 0);
            box.userData = { phase: i * 0.3 };
            this.scene.add(box);
            duality.objectiveTime.push(box);
        }
        
        // 主观时间 - 意识采样
        const subjectiveGeometry = new THREE.SphereGeometry(0.1, 8, 8);
        const subjectiveMaterial = new THREE.MeshPhongMaterial({ color: 0xff6600 });
        
        for (let i = 0; i < 5; i++) {
            const sphere = new THREE.Mesh(subjectiveGeometry, subjectiveMaterial.clone());
            sphere.position.set(i * 2 - 4, 2, 0);
            sphere.userData = { phase: i * 0.5 };
            this.scene.add(sphere);
            duality.subjectiveTime.push(sphere);
        }
        
        this.currentAnimation = duality;
        return duality;
    }
    
    updateTimeDuality() {
        if (!this.currentAnimation) return;
        
        const time = Date.now() * 0.001;
        this.currentAnimation.time = time;
        
        // 更新客观时间
        this.currentAnimation.objectiveTime.forEach((box, index) => {
            const phase = time * 2 + box.userData.phase;
            box.position.y = Math.sin(phase) * 2;
            box.rotation.x = phase;
            box.rotation.y = phase * 0.5;
        });
        
        // 更新主观时间（采样）
        this.currentAnimation.subjectiveTime.forEach((sphere, index) => {
            const phase = time * 0.5 + sphere.userData.phase;
            sphere.position.y = 2 + Math.sin(phase) * 1.5;
            sphere.scale.setScalar(0.5 + Math.sin(phase) * 0.3);
        });
    }
    
    // 哥白尼计划验证动画
    createCopernicusPlanAnimation() {
        this.clearScene();
        
        const copernicus = {
            predictions: [],
            experiments: [],
            comparisons: [],
            time: 0,
            update: () => this.updateCopernicusPlan()
        };
        
        // 创建预测值柱状图
        const predictionData = [
            { name: '希格斯质量', value: 125.3, color: 0x0066ff },
            { name: '质子-中子质量差', value: 1.29, color: 0x00ff66 },
            { name: '电子反常磁矩', value: 0.00116, color: 0xff6600 },
            { name: 'CMB谱指数', value: 0.9642, color: 0xff0066 }
        ];
        
        predictionData.forEach((data, index) => {
            const geometry = new THREE.BoxGeometry(0.5, data.value * 0.1, 0.5);
            const material = new THREE.MeshPhongMaterial({ color: data.color });
            const bar = new THREE.Mesh(geometry, material);
            bar.position.set(index * 2 - 3, data.value * 0.05, 0);
            bar.userData = { data: data };
            this.scene.add(bar);
            copernicus.predictions.push(bar);
        });
        
        this.currentAnimation = copernicus;
        return copernicus;
    }
    
    updateCopernicusPlan() {
        if (!this.currentAnimation) return;
        
        const time = Date.now() * 0.001;
        this.currentAnimation.time = time;
        
        // 更新柱状图动画
        this.currentAnimation.predictions.forEach((bar, index) => {
            const phase = time * 2 + index * 0.5;
            bar.rotation.y = Math.sin(phase) * 0.1;
            bar.scale.y = 1 + Math.sin(phase) * 0.1;
        });
    }
    
    clearScene() {
        // 清除现有动画对象
        if (this.currentAnimation) {
            if (this.currentAnimation.nodes) {
                this.currentAnimation.nodes.forEach(node => this.scene.remove(node));
            }
            if (this.currentAnimation.particles) {
                this.currentAnimation.particles.forEach(particle => this.scene.remove(particle));
            }
            if (this.currentAnimation.connections) {
                this.currentAnimation.connections.forEach(connection => this.scene.remove(connection));
            }
            if (this.currentAnimation.objectiveTime) {
                this.currentAnimation.objectiveTime.forEach(obj => this.scene.remove(obj));
            }
            if (this.currentAnimation.subjectiveTime) {
                this.currentAnimation.subjectiveTime.forEach(obj => this.scene.remove(obj));
            }
            if (this.currentAnimation.predictions) {
                this.currentAnimation.predictions.forEach(pred => this.scene.remove(pred));
            }
        }
        
        // 清除所有子对象
        while (this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }
        
        this.currentAnimation = null;
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.controls) {
            this.controls.dispose();
        }
    }
}

// 导出供外部使用
window.QSDTAnimationSystem = QSDTAnimationSystem;
