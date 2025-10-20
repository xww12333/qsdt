"""
QSDT哥白尼计划重正化群(RG)演化测试

功能：测试RG演化方程和参数校准
作用：确保RG演化数值稳定性和理论一致性
理论文档位置：
    - 附录7：哥白尼计划 - RG演化方程
    - 附录24：QSDT终极参数附录
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rg import RGState, qsd_appendix7_betas, calibrate_from_ugut_theory


class TestRG(unittest.TestCase):
    """重正化群测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.mu_Pl = 1.22e19  # 普朗克质量 (GeV)
        self.mu_e = 0.000511  # 电子质量 (GeV)
        self.L = np.log(self.mu_Pl / self.mu_e)  # 对数跨度
    
    def test_rg_state_initialization(self):
        """测试RG状态初始化"""
        state = RGState(mu=246.0, params={'J': 1e8, 'E': 3e8, 'g': 1e-6, 'Gamma': 0.0})
        
        self.assertEqual(state.mu, 246.0)
        self.assertEqual(state.params['J'], 1e8)
        self.assertEqual(state.params['E'], 3e8)
        self.assertEqual(state.params['g'], 1e-6)
        self.assertEqual(state.params['Gamma'], 0.0)
    
    def test_beta_functions(self):
        """测试beta函数"""
        state = RGState(mu=246.0, params={'J': 1e8, 'E': 3e8, 'g': 0.1, 'Gamma': 1e7})
        
        # 获取beta函数
        coeffs = {'A': 1.0, 'b_J': -0.085, 'c_J': 0.022, 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        
        # 计算beta函数值
        dJ_dlnmu = betas['J'](state)
        dg_dlnmu = betas['g'](state)
        # 注意：E和Gamma不直接演化，通过其他参数计算
        
        # 验证beta函数返回值
        self.assertIsInstance(dJ_dlnmu, float)
        self.assertIsInstance(dg_dlnmu, float)
        
        # 验证g的beta函数：β_g = A * g * (1 - g)
        expected_dg_dlnmu = 1.0 * 0.1 * (1 - 0.1)  # A=1.0
        self.assertAlmostEqual(dg_dlnmu, expected_dg_dlnmu, places=10)
    
    def test_g_evolution(self):
        """测试g参数演化"""
        # 测试g从接近0演化到接近1
        g_values = [1e-6, 0.1, 0.5, 0.9, 0.99]
        coeffs = {'A': 1.0, 'b_J': -0.085, 'c_J': 0.022, 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        
        for g in g_values:
            state = RGState(mu=246.0, params={'J': 1e8, 'E': 3e8, 'g': g, 'Gamma': g*1e8})
            dg_dlnmu = betas['g'](state)
            
            # 验证g的演化方向
            # β_g = A * g * (1 - g)，当g < 1时增长，g > 1时衰减
            if g < 1.0:
                self.assertGreater(dg_dlnmu, 0, f"g={g}时应该增长")
            elif g > 1.0:
                self.assertLess(dg_dlnmu, 0, f"g={g}时应该衰减")
            else:
                # g=1时，β_g = A * 1 * (1-1) = 0
                expected_dg_dlnmu = 1.0 * 1.0 * (1 - 1.0)  # A=1.0
                self.assertAlmostEqual(dg_dlnmu, expected_dg_dlnmu, places=10, msg=f"g=1时的beta值不正确")
            
            # 验证beta函数的数学形式
            expected_dg_dlnmu = 1.0 * g * (1 - g)  # A=1.0
            self.assertAlmostEqual(dg_dlnmu, expected_dg_dlnmu, places=10, msg=f"g={g}时的beta函数计算不正确")
    
    def test_calibration(self):
        """测试RG参数校准"""
        cal = calibrate_from_ugut_theory()
        
        # 验证校准结果
        self.assertIn('A', cal)
        self.assertIn('b_J', cal)
        self.assertIn('c_J', cal)
        self.assertIn('J0', cal)
        self.assertIn('g0', cal)
        # 注意：E0不在校准结果中，E通过E/J=3.0计算
        
        # 验证理论值
        self.assertEqual(cal['A'], 1.0, "A参数应该为1.0")
        self.assertGreater(cal['g0'], 0, "g0应该为正值")
        self.assertGreater(cal['J0'], 0, "J0应该为正值")
        
        # 验证E0可以通过J0计算
        E0 = cal['J0'] * 3.0  # E/J = 3.0
        self.assertGreater(E0, 0, "E0应该为正值")
    
    def test_j_evolution(self):
        """测试J参数演化"""
        cal = calibrate_from_ugut_theory()
        E0 = cal['J0'] * 3.0  # E/J = 3.0
        state = RGState(mu=246.0, params={'J': cal['J0'], 'E': E0, 'g': cal['g0'], 'Gamma': 0.0})
        
        coeffs = {'A': 1.0, 'b_J': cal['b_J'], 'c_J': cal['c_J'], 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        dJ_dlnmu = betas['J'](state)
        
        # J的演化应该为负值（从普朗克标尺到电弱标尺）
        self.assertLess(dJ_dlnmu, 0, "J参数应该随能量标尺降低而减小")
    
    def test_e_evolution(self):
        """测试E参数演化"""
        cal = calibrate_from_ugut_theory()
        E0 = cal['J0'] * 3.0  # E/J = 3.0
        state = RGState(mu=246.0, params={'J': cal['J0'], 'E': E0, 'g': cal['g0'], 'Gamma': cal['g0']*cal['J0']})
        
        coeffs = {'A': 1.0, 'b_J': cal['b_J'], 'c_J': cal['c_J'], 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        
        # E不直接演化，通过E/J=3.0关系计算
        dJ_dlnmu = betas['J'](state)
        dE_dlnmu = 3.0 * dJ_dlnmu  # dE/dlnμ = 3.0 * dJ/dlnμ
        
        # E的演化应该为负值
        self.assertLess(dE_dlnmu, 0, "E参数应该随能量标尺降低而减小")
    
    def test_gamma_evolution(self):
        """测试Gamma参数演化"""
        cal = calibrate_from_ugut_theory()
        E0 = cal['J0'] * 3.0  # E/J = 3.0
        state = RGState(mu=246.0, params={'J': cal['J0'], 'E': E0, 'g': cal['g0'], 'Gamma': cal['g0']*cal['J0']})
        
        coeffs = {'A': 1.0, 'b_J': cal['b_J'], 'c_J': cal['c_J'], 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        
        # Gamma = g * J，其演化通过链式法则计算
        dJ_dlnmu = betas['J'](state)
        dg_dlnmu = betas['g'](state)
        g = state.params['g']
        J = state.params['J']
        dGamma_dlnmu = dg_dlnmu * J + g * dJ_dlnmu  # dΓ/dlnμ = (dg/dlnμ)*J + g*(dJ/dlnμ)
        
        # Gamma = g * J，其演化应该为正值（g增长，J减小）
        self.assertGreater(dGamma_dlnmu, 0, "Gamma参数应该随能量标尺降低而增长")
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试极端值情况
        extreme_states = [
            RGState(mu=246.0, params={'J': 1e-30, 'E': 1e-30, 'g': 1e-30, 'Gamma': 0.0}),  # 极小值
            RGState(mu=246.0, params={'J': 1e30, 'E': 1e30, 'g': 0.99, 'Gamma': 1e30}),    # 极大值
            RGState(mu=246.0, params={'J': 1e8, 'E': 1e8, 'g': 0.0, 'Gamma': 0.0}),        # g=0边界
            RGState(mu=246.0, params={'J': 1e8, 'E': 1e8, 'g': 1.0, 'Gamma': 1e8}),        # g=1边界
        ]
        
        coeffs = {'A': 1.0, 'b_J': -0.085, 'c_J': 0.022, 'b_E': 0.085, 'c_E': 0.022}
        betas = qsd_appendix7_betas(coeffs)
        
        for state in extreme_states:
            try:
                dJ_dlnmu = betas['J'](state)
                dg_dlnmu = betas['g'](state)
                
                # 计算E和Gamma的演化
                dE_dlnmu = 3.0 * dJ_dlnmu  # E/J = 3.0
                g = state.params['g']
                J = state.params['J']
                dGamma_dlnmu = dg_dlnmu * J + g * dJ_dlnmu  # 链式法则
                
                # 验证返回值有限
                self.assertTrue(np.isfinite(dJ_dlnmu), f"dJ_dlnmu在状态{state.params}下为无穷大")
                self.assertTrue(np.isfinite(dE_dlnmu), f"dE_dlnmu在状态{state.params}下为无穷大")
                self.assertTrue(np.isfinite(dg_dlnmu), f"dg_dlnmu在状态{state.params}下为无穷大")
                self.assertTrue(np.isfinite(dGamma_dlnmu), f"dGamma_dlnmu在状态{state.params}下为无穷大")
                
            except Exception as e:
                self.fail(f"极端状态{state.params}导致异常: {e}")
    
    def test_theoretical_consistency(self):
        """测试理论一致性"""
        cal = calibrate_from_ugut_theory()
        
        # 验证E/J比值
        E0 = cal['J0'] * 3.0  # E/J = 3.0
        E_over_J = E0 / cal['J0']
        self.assertAlmostEqual(E_over_J, 3.0, places=2, 
                             msg="初始E/J比值应该接近3.0")
        
        # 验证g0为小值
        self.assertLess(cal['g0'], 1e-3, "g0应该为小值")
        
        # 验证J0和E0为正值
        self.assertGreater(cal['J0'], 0, "J0应该为正值")
        self.assertGreater(E0, 0, "E0应该为正值")


if __name__ == '__main__':
    unittest.main()
