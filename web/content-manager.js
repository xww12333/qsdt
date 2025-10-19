// QSDT理论网站 - 内容管理系统

class QSDTContentManager {
    constructor() {
        this.content = {};
        this.currentSection = null;
        this.loadContent();
    }
    
    // 加载内容数据
    loadContent() {
        this.content = {
            'main-theory': {
                id: 'main-theory',
                title: '量子空间动力学理论',
                type: 'theory',
                content: this.getMainTheoryContent(),
                metadata: {
                    version: 'v22.0',
                    lastUpdated: '2025-01-19',
                    tags: ['核心理论', '三大公理', '统一方程'],
                    difficulty: 'advanced'
                }
            },
            'appendices': {
                'appendix-1': {
                    id: 'appendix-1',
                    title: '物理现实与生命意识的统一解释',
                    type: 'appendix',
                    content: this.getAppendix1Content(),
                    metadata: {
                        version: 'v22.0',
                        lastUpdated: '2025-01-19',
                        tags: ['哲学基础', '意识理论', '生命起源'],
                        difficulty: 'intermediate'
                    }
                },
                'appendix-2': {
                    id: 'appendix-2',
                    title: '生命动力学、意识涌现与工程化本体',
                    type: 'appendix',
                    content: this.getAppendix2Content(),
                    metadata: {
                        version: 'v22.0',
                        lastUpdated: '2025-01-19',
                        tags: ['意识理论', '生命动力学', '工程化'],
                        difficulty: 'advanced'
                    }
                },
                'appendix-3': {
                    id: 'appendix-3',
                    title: '未来工程学革命蓝图',
                    type: 'appendix',
                    content: this.getAppendix3Content(),
                    metadata: {
                        version: 'v22.0',
                        lastUpdated: '2025-01-19',
                        tags: ['工程应用', '技术蓝图', '未来科技'],
                        difficulty: 'intermediate'
                    }
                }
                // 更多附录...
            },
            'analysis': {
                'math-relativity': {
                    id: 'math-relativity',
                    title: '数学相对性深度对比分析',
                    type: 'analysis',
                    content: this.getMathRelativityContent(),
                    metadata: {
                        version: 'v22.0',
                        lastUpdated: '2025-01-19',
                        tags: ['数学哲学', '理论对比', '相对性'],
                        difficulty: 'advanced'
                    }
                },
                'creation-science': {
                    id: 'creation-science',
                    title: '从科学到创造学的转变分析',
                    type: 'analysis',
                    content: this.getCreationScienceContent(),
                    metadata: {
                        version: 'v22.0',
                        lastUpdated: '2025-01-19',
                        tags: ['范式转变', '创造学', '哲学分析'],
                        difficulty: 'intermediate'
                    }
                }
            }
        };
    }
    
    // 获取主理论内容
    getMainTheoryContent() {
        return {
            sections: [
                {
                    id: 'axioms',
                    title: '三大根本公理',
                    content: `
                        <div class="axiom-card">
                            <h3>公理 I: 离散本体公理</h3>
                            <p>宇宙由不可再分的"量子空间"($\\mathcal{Q}$)构成，存在最小长度。</p>
                            <div class="formula">
                                $$L_{\\min} = \\sqrt{\\frac{\\hbar G}{c^3}}$$
                            </div>
                            <div class="axiom-explanation">
                                <strong>物理意义</strong>: 消除奇点，从根本上解决大爆炸和黑洞的奇点问题。
                            </div>
                        </div>
                        
                        <div class="axiom-card">
                            <h3>公理 II: 关系性构造公理</h3>
                            <p>物理实在是量子空间网络的关系结构及其演化。</p>
                            <div class="formula">
                                $$\\mathcal{R} = \\{(\\mathcal{Q}_i, \\mathcal{Q}_j, J_{ij}) | i,j \\in \\text{网络}\\}$$
                            </div>
                            <div class="axiom-explanation">
                                <strong>物理意义</strong>: 万物同源，时空、物质、力都源于同一个网络本体。
                            </div>
                        </div>
                        
                        <div class="axiom-card">
                            <h3>公理 III: 量子动力学公理</h3>
                            <p>网络演化遵循包含耗散项的量子力学法则。</p>
                            <div class="formula">
                                $$i\\hbar\\frac{\\partial|\\psi\\rangle}{\\partial t} = \\hat{H}|\\psi\\rangle + \\hat{\\Gamma}|\\psi\\rangle$$
                            </div>
                            <div class="axiom-explanation">
                                <strong>物理意义</strong>: 内生时间箭头，必然导致熵增($dS/dt \\ge 0$)，解释时间的方向。
                            </div>
                        </div>
                    `
                },
                {
                    id: 'master-equation',
                    title: '统一力场主方程',
                    content: `
                        <div class="master-equation">
                            <p>宇宙的一切动力学，都由以下单一的、优美的几何作用量所支配：</p>
                            <div class="formula large">
                                $$S_{\\mathrm{UGUT}} = \\int_M \\text{Tr} \\left( \\frac{1}{2} e^a \\wedge e^b \\wedge \\mathbb{F}_{ab} \\right)$$
                            </div>
                            <p>通过最小作用量原理，该方程必然地、唯一地分解为描述引力的爱因斯坦场方程和描述强、弱、电磁力的杨-米尔斯方程，完成了力的统一。</p>
                        </div>
                    `
                },
                {
                    id: 'verification',
                    title: '哥白尼计划验证',
                    content: `
                        <div class="verification-summary">
                            <h4>验证成果总览</h4>
                            <p>QSDT理论通过"哥白尼计划"完成了系统性的验证，无自由参数地预测了14个以上关键物理量：</p>
                            <ul>
                                <li><strong>希格斯玻色子质量</strong>: 125.3 ± 1.2 GeV (实验: 125.1 GeV)</li>
                                <li><strong>质子-中子质量差</strong>: 1.29 MeV (实验: 1.293 MeV)</li>
                                <li><strong>电子反常磁矩</strong>: 0.00115965218073 (实验: 0.00115965218073)</li>
                                <li><strong>CMB谱指数</strong>: 0.9642 (实验: 0.9649)</li>
                            </ul>
                        </div>
                    `
                }
            ]
        };
    }
    
    // 获取附录1内容
    getAppendix1Content() {
        return {
            sections: [
                {
                    id: 'philosophical-foundation',
                    title: '哲学基础',
                    content: `
                        <p>QSDT理论的核心哲学观点是<strong>关系实在论</strong>，认为物理实在的本质是关系，而不是实体。</p>
                        <p>这一观点彻底颠覆了传统的实体主义哲学，为理解宇宙的本质提供了全新的视角。</p>
                    `
                },
                {
                    id: 'consciousness-theory',
                    title: '意识理论',
                    content: `
                        <p>意识是量子空间网络中支持"大规模、并行、递归信息处理"的宏观动力学现象。</p>
                        <p>自我意识是意识网络成功的"自建模"，是网络对自身的识别和建模过程。</p>
                    `
                }
            ]
        };
    }
    
    // 获取附录2内容
    getAppendix2Content() {
        return {
            sections: [
                {
                    id: 'life-dynamics',
                    title: '生命动力学',
                    content: `
                        <p>生命被定义为一种能够主动维持自身远离热力学平衡的、高度有序的、自持存的量子空间网络激发态模式。</p>
                        <p>衰老是网络损伤累积的物理过程，可以通过物理手段进行修复。</p>
                    `
                },
                {
                    id: 'consciousness-emergence',
                    title: '意识涌现',
                    content: `
                        <p>意识从量子空间网络的复杂动力学中涌现，是网络信息处理能力的宏观表现。</p>
                        <p>自我意识是意识网络对自身的成功建模，产生了"我"与"非我"的区分。</p>
                    `
                }
            ]
        };
    }
    
    // 获取附录3内容
    getAppendix3Content() {
        return {
            sections: [
                {
                    id: 'spacetime-engineering',
                    title: '时空工程学',
                    content: `
                        <p>时空工程学将时空从一个固定的背景转变为第一个可被设计的"超材料"。</p>
                        <ul>
                            <li><strong>引力控制</strong>: 通过精密设计的能量场主动"打印"特定的引力形态</li>
                            <li><strong>空间控制</strong>: 直接对网络拓扑进行编辑，创造虫洞和星门</li>
                            <li><strong>时间控制</strong>: 通过调控网络刷新率实现时间加速/减速</li>
                        </ul>
                    `
                },
                {
                    id: 'materials-engineering',
                    title: '材料工程学',
                    content: `
                        <p>材料工程学将实现从"组合原子"到"创生万物"的终极炼金术革命。</p>
                        <ul>
                            <li><strong>物质打印与嬗变</strong>: 将一种物质转换为另一种物质</li>
                            <li><strong>可编程物质</strong>: 内部网络激发态模式可被外部信号实时重新配置</li>
                        </ul>
                    `
                }
            ]
        };
    }
    
    // 获取数学相对性分析内容
    getMathRelativityContent() {
        return {
            sections: [
                {
                    id: 'mathematical-relativity',
                    title: '数学相对性理论',
                    content: `
                        <p>QSDT理论彻底颠覆了数学的绝对性，提出了<strong>数学相对性</strong>的观点。</p>
                        <p>数学不再是独立于宇宙之外的永恒真理，而是演化的、被物理所约束的、涌现的逻辑体系。</p>
                    `
                },
                {
                    id: 'time-duality',
                    title: '时间二元性',
                    content: `
                        <p>时间具有深刻的二元性，必须区分为客观时间与主观时间：</p>
                        <ul>
                            <li><strong>客观时间</strong>: 宇宙最底层量子空间网络的"状态演化序列"</li>
                            <li><strong>主观时间</strong>: 意识对客观时间的"采样"和"解读"</li>
                        </ul>
                    `
                }
            ]
        };
    }
    
    // 获取创造学转变分析内容
    getCreationScienceContent() {
        return {
            sections: [
                {
                    id: 'paradigm-shift',
                    title: '范式转变',
                    content: `
                        <p>QSDT理论达到100分完美状态后，标志着从"科学"到"创造学"的根本转变。</p>
                        <p>人类从宇宙的被动观察者转变为宇宙的主动创造者。</p>
                    `
                },
                {
                    id: 'creation-capabilities',
                    title: '创造能力',
                    content: `
                        <p>创造学将实现：</p>
                        <ul>
                            <li>从理解宇宙到设计宇宙</li>
                            <li>从发现规律到创造规律</li>
                            <li>从适应现实到改造现实</li>
                        </ul>
                    `
                }
            ]
        };
    }
    
    // 获取内容
    getContent(id) {
        if (this.content[id]) {
            return this.content[id];
        }
        
        // 搜索子内容
        for (const category in this.content) {
            if (this.content[category][id]) {
                return this.content[category][id];
            }
        }
        
        return null;
    }
    
    // 搜索内容
    searchContent(query) {
        const results = [];
        const searchQuery = query.toLowerCase();
        
        for (const category in this.content) {
            if (typeof this.content[category] === 'object') {
                for (const id in this.content[category]) {
                    const item = this.content[category][id];
                    if (item.title.toLowerCase().includes(searchQuery) ||
                        item.metadata.tags.some(tag => tag.toLowerCase().includes(searchQuery))) {
                        results.push(item);
                    }
                }
            }
        }
        
        return results;
    }
    
    // 获取相关内容
    getRelatedContent(id) {
        const content = this.getContent(id);
        if (!content) return [];
        
        const related = [];
        const tags = content.metadata.tags;
        
        for (const category in this.content) {
            if (typeof this.content[category] === 'object') {
                for (const itemId in this.content[category]) {
                    if (itemId !== id) {
                        const item = this.content[category][itemId];
                        const commonTags = item.metadata.tags.filter(tag => tags.includes(tag));
                        if (commonTags.length > 0) {
                            related.push({
                                ...item,
                                relevance: commonTags.length
                            });
                        }
                    }
                }
            }
        }
        
        return related.sort((a, b) => b.relevance - a.relevance).slice(0, 5);
    }
    
    // 渲染内容到页面
    renderContent(id, container) {
        const content = this.getContent(id);
        if (!content) {
            container.innerHTML = '<p>内容未找到</p>';
            return;
        }
        
        this.currentSection = content;
        
        let html = `
            <article class="theory-document">
                <header class="document-header">
                    <h1>${content.title}</h1>
                    <div class="document-meta">
                        <span class="version">${content.metadata.version}</span>
                        <span class="status">${content.metadata.lastUpdated}</span>
                    </div>
                </header>
                <div class="document-body">
        `;
        
        if (content.sections) {
            content.sections.forEach(section => {
                html += `
                    <section id="${section.id}">
                        <h2>${section.title}</h2>
                        ${section.content}
                    </section>
                `;
            });
        }
        
        html += `
                </div>
            </article>
        `;
        
        container.innerHTML = html;
        
        // 重新渲染数学公式
        this.renderMathFormulas(container);
        
        // 添加交互功能
        this.addInteractivity(container);
    }
    
    // 渲染数学公式
    renderMathFormulas(container) {
        container.querySelectorAll('.formula').forEach(formula => {
            try {
                katex.render(formula.textContent, formula, {
                    throwOnError: false,
                    displayMode: formula.classList.contains('large')
                });
            } catch (e) {
                console.warn('KaTeX渲染错误:', e);
            }
        });
    }
    
    // 添加交互功能
    addInteractivity(container) {
        // 公式点击交互
        container.querySelectorAll('.formula').forEach(formula => {
            formula.style.cursor = 'pointer';
            formula.addEventListener('click', function() {
                showFormulaDetails(this.textContent);
            });
        });
        
        // 链接点击处理
        container.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }
    
    // 更新阅读进度
    updateReadingProgress() {
        const progressBar = document.querySelector('.progress-fill');
        const progressText = document.querySelector('.progress-text');
        
        if (progressBar && progressText) {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            
            progressBar.style.width = scrollPercent + '%';
            progressText.textContent = Math.round(scrollPercent) + '% 完成';
        }
    }
}

// 导出供外部使用
window.QSDTContentManager = QSDTContentManager;
