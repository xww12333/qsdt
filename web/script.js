// QSDT理论网站JavaScript功能

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    initializeAnimations();
    initializeTheoryReader();
    initializeVerificationCards();
    initializeScrollEffects();
});

// 导航功能
function initializeNavigation() {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
    }
    
    // 平滑滚动到锚点
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// 动画系统初始化
function initializeAnimations() {
    const animationSelect = document.getElementById('animationSelect');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const resetBtn = document.getElementById('resetBtn');
    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    const animationCanvas = document.getElementById('animationCanvas');
    const animationTitle = document.getElementById('animationTitle');
    const animationDescription = document.getElementById('animationDescription');
    
    // 动画数据
    const animations = {
        'quantum-network': {
            title: '量子空间网络演化',
            description: '展示离散的量子空间如何形成动态网络，以及网络如何演化产生我们所观察到的物理现象。',
            duration: 120000, // 2分钟
            parameters: ['density', 'coupling', 'fluctuation']
        },
        'symmetry-breaking': {
            title: '对称性级联破缺',
            description: '展示宇宙从高对称性状态到低对称性状态的演化过程，以及四种基本力的分离。',
            duration: 90000, // 1.5分钟
            parameters: ['temperature', 'energy', 'phase']
        },
        'particle-emergence': {
            title: '粒子涌现过程',
            description: '展示粒子如何从量子空间网络中作为激发态涌现出来。',
            duration: 60000, // 1分钟
            parameters: ['excitation', 'topology', 'mass']
        },
        'time-duality': {
            title: '时间二元性演示',
            description: '展示客观时间与主观时间的区别，以及时间扭曲现象。',
            duration: 90000, // 1.5分钟
            parameters: ['objective', 'subjective', 'distortion']
        },
        'copernicus-plan': {
            title: '哥白尼计划验证',
            description: '展示QSDT理论预测与实验测量的对比验证过程。',
            duration: 180000, // 3分钟
            parameters: ['prediction', 'experiment', 'accuracy']
        }
    };
    
    let currentAnimation = null;
    let animationInterval = null;
    let isPlaying = false;
    let currentSpeed = 1.0;
    
    // 动画选择
    if (animationSelect) {
        animationSelect.addEventListener('change', function() {
            const animationId = this.value;
            const animation = animations[animationId];
            
            if (animation) {
                animationTitle.textContent = animation.title;
                animationDescription.textContent = animation.description;
                currentAnimation = animation;
                
                // 更新参数控制
                updateParameterControls(animation.parameters);
                
                // 重置动画
                resetAnimation();
            }
        });
    }
    
    // 播放控制
    if (playBtn) {
        playBtn.addEventListener('click', function() {
            if (!isPlaying) {
                startAnimation();
                isPlaying = true;
                playBtn.textContent = '暂停';
                pauseBtn.textContent = '暂停';
            }
        });
    }
    
    if (pauseBtn) {
        pauseBtn.addEventListener('click', function() {
            if (isPlaying) {
                pauseAnimation();
                isPlaying = false;
                playBtn.textContent = '播放';
                pauseBtn.textContent = '播放';
            }
        });
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            resetAnimation();
            isPlaying = false;
            playBtn.textContent = '播放';
            pauseBtn.textContent = '播放';
        });
    }
    
    // 速度控制
    if (speedSlider && speedValue) {
        speedSlider.addEventListener('input', function() {
            currentSpeed = parseFloat(this.value);
            speedValue.textContent = currentSpeed.toFixed(1) + 'x';
            
            if (isPlaying) {
                pauseAnimation();
                startAnimation();
            }
        });
    }
    
    // 动画控制函数
    function startAnimation() {
        if (!currentAnimation) return;
        
        const frameDuration = 1000 / (30 * currentSpeed); // 30fps
        let progress = 0;
        
        animationInterval = setInterval(() => {
            progress += frameDuration / currentAnimation.duration;
            
            if (progress >= 1) {
                progress = 0;
            }
            
            updateAnimationFrame(progress);
        }, frameDuration);
    }
    
    function pauseAnimation() {
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
    }
    
    function resetAnimation() {
        pauseAnimation();
        updateAnimationFrame(0);
    }
    
    function updateAnimationFrame(progress) {
        // 这里将集成Three.js来渲染3D动画
        // 目前使用CSS动画作为占位符
        const canvas = animationCanvas;
        if (canvas) {
            canvas.style.background = `linear-gradient(${progress * 360}deg, 
                rgba(0, 102, 255, ${0.1 + progress * 0.3}), 
                rgba(255, 102, 0, ${0.1 + progress * 0.2}))`;
        }
    }
    
    function updateParameterControls(parameters) {
        // 更新参数控制面板
        const parameterContainer = document.querySelector('.animation-parameters');
        if (parameterContainer) {
            parameterContainer.innerHTML = '<h4>可调参数</h4>';
            
            parameters.forEach(param => {
                const paramGroup = document.createElement('div');
                paramGroup.className = 'parameter-group';
                paramGroup.innerHTML = `
                    <label>${getParameterLabel(param)}:</label>
                    <input type="range" id="${param}Slider" min="0" max="1" step="0.01" value="0.5">
                `;
                parameterContainer.appendChild(paramGroup);
            });
        }
    }
    
    function getParameterLabel(param) {
        const labels = {
            'density': '网络密度',
            'coupling': '耦合强度',
            'fluctuation': '量子涨落',
            'temperature': '温度',
            'energy': '能量',
            'phase': '相位',
            'excitation': '激发强度',
            'topology': '拓扑结构',
            'mass': '质量',
            'objective': '客观时间',
            'subjective': '主观时间',
            'distortion': '时间扭曲',
            'prediction': '预测精度',
            'experiment': '实验精度',
            'accuracy': '验证精度'
        };
        return labels[param] || param;
    }
}

// 理论阅读器功能
function initializeTheoryReader() {
    // 阅读进度跟踪
    const progressBar = document.querySelector('.progress-fill');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar && progressText) {
        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            
            progressBar.style.width = scrollPercent + '%';
            progressText.textContent = Math.round(scrollPercent) + '% 完成';
        });
    }
    
    // 书签功能
    const bookmarkList = document.querySelector('.bookmark-list');
    if (bookmarkList) {
        // 添加书签
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'b') {
                e.preventDefault();
                addBookmark();
            }
        });
    }
    
    // 数学公式交互
    initializeMathFormulas();
}

// 数学公式交互功能
function initializeMathFormulas() {
    // 渲染数学公式
    document.querySelectorAll('.formula').forEach(formula => {
        try {
            katex.render(formula.textContent, formula, {
                throwOnError: false,
                displayMode: formula.classList.contains('large')
            });
        } catch (e) {
            console.warn('KaTeX渲染错误:', e);
        }
    });
    
    // 公式点击交互
    document.querySelectorAll('.formula').forEach(formula => {
        formula.style.cursor = 'pointer';
        formula.addEventListener('click', function() {
            showFormulaDetails(this.textContent);
        });
    });
}

// 显示公式详情
function showFormulaDetails(formula) {
    // 创建模态框显示公式详情
    const modal = document.createElement('div');
    modal.className = 'formula-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <h3>公式详情</h3>
            <div class="formula-display">${formula}</div>
            <div class="formula-explanation">
                <p>这是QSDT理论中的核心公式，描述了量子空间网络的动力学行为。</p>
                <p>点击公式中的变量可以查看其物理意义和数学定义。</p>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭模态框
    modal.querySelector('.close').addEventListener('click', function() {
        document.body.removeChild(modal);
    });
    
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
}

// 添加书签
function addBookmark() {
    const currentSection = getCurrentSection();
    if (currentSection) {
        const bookmarkList = document.querySelector('.bookmark-list');
        const bookmark = document.createElement('li');
        bookmark.innerHTML = `<a href="#${currentSection.id}">${currentSection.title}</a>`;
        bookmarkList.appendChild(bookmark);
        
        // 显示添加成功提示
        showNotification('书签已添加', 'success');
    }
}

// 获取当前章节
function getCurrentSection() {
    const sections = document.querySelectorAll('h1, h2, h3');
    let currentSection = null;
    
    sections.forEach(section => {
        const rect = section.getBoundingClientRect();
        if (rect.top <= 100 && rect.bottom >= 100) {
            currentSection = {
                id: section.id || section.textContent.toLowerCase().replace(/\s+/g, '-'),
                title: section.textContent
            };
        }
    });
    
    return currentSection;
}

// 验证卡片功能
function initializeVerificationCards() {
    const verificationCards = document.querySelectorAll('.verification-card');
    
    verificationCards.forEach(card => {
        card.addEventListener('click', function() {
            const title = this.querySelector('h3').textContent;
            showVerificationDetails(title);
        });
    });
}

// 显示验证详情
function showVerificationDetails(title) {
    const modal = document.createElement('div');
    modal.className = 'verification-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <h3>${title} - 详细验证</h3>
            <div class="verification-details">
                <div class="prediction-method">
                    <h4>QSDT预测方法</h4>
                    <p>从第一性原理出发，通过已校准的贝塔函数计算得出...</p>
                </div>
                <div class="experimental-method">
                    <h4>实验测量方法</h4>
                    <p>通过高能物理实验，使用精密测量设备获得...</p>
                </div>
                <div class="comparison">
                    <h4>对比分析</h4>
                    <p>预测值与实验值高度吻合，误差小于0.2%...</p>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 关闭模态框
    modal.querySelector('.close').addEventListener('click', function() {
        document.body.removeChild(modal);
    });
    
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
}

// 滚动效果
function initializeScrollEffects() {
    // 导航栏滚动效果
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            // 向下滚动，隐藏导航栏
            navbar.style.transform = 'translateY(-100%)';
        } else {
            // 向上滚动，显示导航栏
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScrollTop = scrollTop;
    });
    
    // 元素进入视口动画
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // 观察需要动画的元素
    document.querySelectorAll('.achievement-card, .axiom-card, .verification-card').forEach(card => {
        observer.observe(card);
    });
}

// 显示通知
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // 显示动画
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    // 自动隐藏
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// 工具函数
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 导出函数供外部使用
window.QSDTWebsite = {
    showNotification,
    addBookmark,
    showFormulaDetails,
    showVerificationDetails
};

// 添加CSS样式到页面
const additionalStyles = `
<style>
/* 模态框样式 */
.formula-modal,
.verification-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
}

.modal-content {
    background: var(--gray-900);
    padding: var(--spacing-2xl);
    border-radius: var(--radius-lg);
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    position: relative;
    border: 1px solid var(--gray-700);
}

.close {
    position: absolute;
    top: var(--spacing-md);
    right: var(--spacing-md);
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--gray-400);
}

.close:hover {
    color: var(--white);
}

/* 通知样式 */
.notification {
    position: fixed;
    top: 100px;
    right: var(--spacing-lg);
    background: var(--gray-800);
    color: var(--white);
    padding: var(--spacing-md) var(--spacing-lg);
    border-radius: var(--radius-md);
    border-left: 4px solid var(--primary);
    transform: translateX(100%);
    transition: transform 0.3s ease;
    z-index: 10001;
}

.notification.show {
    transform: translateX(0);
}

.notification-success {
    border-left-color: var(--secondary);
}

/* 动画效果 */
.animate-in {
    animation: slideInUp 0.6s ease-out;
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 导航栏动画 */
.navbar {
    transition: transform 0.3s ease;
}

/* 移动端导航菜单 */
@media (max-width: 768px) {
    .nav-menu {
        position: fixed;
        top: 70px;
        left: -100%;
        width: 100%;
        height: calc(100vh - 70px);
        background: var(--gray-900);
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        padding-top: var(--spacing-2xl);
        transition: left 0.3s ease;
    }
    
    .nav-menu.active {
        left: 0;
    }
    
    .nav-toggle.active span:nth-child(1) {
        transform: rotate(-45deg) translate(-5px, 6px);
    }
    
    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }
    
    .nav-toggle.active span:nth-child(3) {
        transform: rotate(45deg) translate(-5px, -6px);
    }
}
</style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);
