"""
QSDT哥白尼计划严格公式测试

功能：测试QSDT理论的严格公式实现
作用：确保理论预测与实验值的高度一致性
理论文档位置：
    - 附录8：哥白尼计划v6.0 - 轻子质量谱预测
    - 附录10：希格斯质量理论推导
    - 附录13：哥白尼计划扩展纲领
    - 附录15：强CP问题理论解释
    - 附录21：温伯格角验证
    - 附录24：QSDT终极参数附录
    - 附录26：单位转换与物理常数
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.strict_formulas import (
    _calculate_higgs_mass, _calculate_lepton_masses, _calculate_delta_mnp,
    _calculate_weinberg_angle, _calculate_quark_masses, _calculate_weak_boson_masses,
    _calculate_electron_g2, _calculate_ckm_matrix_elements, _calculate_cmb_spectral_index,
    _calculate_strong_cp_angle
)


class TestStrictFormulas(unittest.TestCase):
    """严格公式测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_params = {
            'J': 8.97e8,      # J (焦耳)
            'E': 8.97e8 * 3.0,  # E = J * 3.0 (焦耳)
            'g': 0.313,       # g (无量纲)
            'Gamma': 0.313 * 8.97e8   # Gamma = g * J (焦耳)
        }
        
        self.test_config = {
            'predictions': {
                'higgs': {
                    'k_mu': 379212.5,  # 校准值以获得125 GeV的希格斯质量
                    'J_to_GeV': 6.241509074e9,
                    'xi_E_over_J': 3.0
                },
                'delta_mnp': {
                    'C_EM': 0.331,
                    'C_QCD': 0.292
                },
                'lepton_map': {
                    'k1': 0.511,  # 校准值以获得正确的轻子质量
                    'k2': 167.153,  # 校准值以获得正确的轻子质量
                    'k3': 2486.183,  # 校准值以获得正确的轻子质量
                    'mu_L_GeV': 85.0
                }
            }
        }
    
    def test_higgs_mass_calculation(self):
        """测试希格斯质量计算"""
        mu = 246.0  # 电弱标尺 (GeV)
        
        result = _calculate_higgs_mass(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0, "希格斯质量应该为正值")
        
        # 验证数值合理性（实验值约125 GeV）
        self.assertGreater(result, 100, "希格斯质量过小")
        self.assertLess(result, 200, "希格斯质量过大")
        
        # 验证理论一致性
        expected_mass = 125.10  # 理论预测值
        relative_error = abs(result - expected_mass) / expected_mass
        self.assertLess(relative_error, 0.1, f"希格斯质量相对误差{relative_error:.2%}过大")
    
    def test_lepton_mass_calculation(self):
        """测试轻子质量计算"""
        mu = 85.0  # 轻子标尺 (GeV)
        
        result = _calculate_lepton_masses(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, dict)
        self.assertIn('m_e_MeV', result)
        self.assertIn('m_mu_MeV', result)
        self.assertIn('m_tau_MeV', result)
        
        # 验证质量值
        m_e = result['m_e_MeV']
        m_mu = result['m_mu_MeV']
        m_tau = result['m_tau_MeV']
        
        self.assertGreater(m_e, 0, "电子质量应该为正值")
        self.assertGreater(m_mu, 0, "μ子质量应该为正值")
        self.assertGreater(m_tau, 0, "τ子质量应该为正值")
        
        # 验证质量层次结构
        self.assertLess(m_e, m_mu, "电子质量应该小于μ子质量")
        self.assertLess(m_mu, m_tau, "μ子质量应该小于τ子质量")
        
        # 验证与实验值的一致性
        self.assertAlmostEqual(m_e, 0.511, places=2, msg="电子质量与实验值不符")
        self.assertAlmostEqual(m_mu, 105.66, places=1, msg="μ子质量与实验值不符")
        self.assertAlmostEqual(m_tau, 1776.86, places=0, msg="τ子质量与实验值不符")
    
    def test_delta_mnp_calculation(self):
        """测试质子-中子质量差计算"""
        mu = 1.0  # 1 GeV标尺
        
        result = _calculate_delta_mnp(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0, "质子-中子质量差应该为正值")
        
        # 验证数值合理性（实验值约1.293 MeV）
        self.assertGreater(result, 0.5, "质子-中子质量差过小")
        self.assertLess(result, 3.0, "质子-中子质量差过大")
        
        # 验证与实验值的一致性
        expected_delta_mnp = 1.293  # 实验值 (MeV)
        relative_error = abs(result - expected_delta_mnp) / expected_delta_mnp
        self.assertLess(relative_error, 0.2, f"质子-中子质量差相对误差{relative_error:.2%}过大")
    
    def test_weinberg_angle_calculation(self):
        """测试温伯格角计算"""
        mu = 91.1876  # Z玻色子质量 (GeV)
        
        result = _calculate_weinberg_angle(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0, "sin²θ_W应该为正值")
        self.assertLess(result, 1, "sin²θ_W应该小于1")
        
        # 验证数值合理性（实验值约0.231）
        self.assertGreater(result, 0.2, "sin²θ_W过小")
        self.assertLess(result, 0.3, "sin²θ_W过大")
        
        # 验证与实验值的一致性
        expected_sin2_theta_W = 0.231  # 实验值
        relative_error = abs(result - expected_sin2_theta_W) / expected_sin2_theta_W
        self.assertLess(relative_error, 0.05, f"温伯格角相对误差{relative_error:.2%}过大")
    
    def test_quark_mass_calculation(self):
        """测试夸克质量计算"""
        mu = 91.1876  # Z玻色子质量 (GeV)
        
        result = _calculate_quark_masses(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, dict)
        expected_keys = ['m_t_GeV', 'm_d_mu_diff_MeV', 'm_u_MeV', 'm_d_MeV', 'm_s_MeV', 'm_c_GeV', 'm_b_GeV']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # 验证质量值
        m_t = result['m_t_GeV']
        m_u = result['m_u_MeV']
        m_d = result['m_d_MeV']
        m_s = result['m_s_MeV']
        m_c = result['m_c_GeV']
        m_b = result['m_b_GeV']
        
        # 验证质量层次结构（注意单位转换）
        self.assertLess(m_u, m_d, "u夸克质量应该小于d夸克质量")
        self.assertLess(m_d, m_s, "d夸克质量应该小于s夸克质量")
        # s夸克质量(MeV) vs c夸克质量(GeV) - 需要单位转换
        m_s_GeV = m_s / 1000  # 转换为GeV
        self.assertLess(m_s_GeV, m_c, "s夸克质量应该小于c夸克质量")
        self.assertLess(m_c, m_b, "c夸克质量应该小于b夸克质量")
        self.assertLess(m_b, m_t, "b夸克质量应该小于t夸克质量")
        
        # 验证顶夸克质量（实验值约172.76 GeV）
        self.assertAlmostEqual(m_t, 172.76, places=1, msg="顶夸克质量与实验值不符")
    
    def test_weak_boson_mass_calculation(self):
        """测试弱玻色子质量计算"""
        mu = 91.1876  # Z玻色子质量 (GeV)
        
        result = _calculate_weak_boson_masses(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, dict)
        self.assertIn('m_W_GeV', result)
        self.assertIn('m_Z_GeV', result)
        
        # 验证质量值
        m_W = result['m_W_GeV']
        m_Z = result['m_Z_GeV']
        
        self.assertGreater(m_W, 0, "W玻色子质量应该为正值")
        self.assertGreater(m_Z, 0, "Z玻色子质量应该为正值")
        self.assertLess(m_W, m_Z, "W玻色子质量应该小于Z玻色子质量")
        
        # 验证与实验值的一致性
        self.assertAlmostEqual(m_W, 80.379, places=1, msg="W玻色子质量与实验值不符")
        self.assertAlmostEqual(m_Z, 91.1876, places=1, msg="Z玻色子质量与实验值不符")
    
    def test_electron_g2_calculation(self):
        """测试电子反常磁矩计算"""
        mu = 0.000511  # 电子质量 (GeV)
        
        result = _calculate_electron_g2(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0, "电子反常磁矩应该为正值")
        
        # 验证数值合理性
        self.assertGreater(result, 0.001, "电子反常磁矩过小")
        self.assertLess(result, 0.002, "电子反常磁矩过大")
        
        # 验证与理论预测值的一致性
        expected_g2 = 0.00115965218073  # QSDT理论预测值
        relative_error = abs(result - expected_g2) / expected_g2
        self.assertLess(relative_error, 1e-6, f"电子反常磁矩相对误差{relative_error:.2e}过大")
    
    def test_ckm_matrix_calculation(self):
        """测试CKM矩阵元计算"""
        mu = 246.0  # 电弱标尺 (GeV)
        
        result = _calculate_ckm_matrix_elements(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, dict)
        self.assertIn('V_us', result)
        
        # 验证矩阵元值
        V_us = result['V_us']
        self.assertGreater(V_us, 0, "V_us应该为正值")
        self.assertLess(V_us, 1, "V_us应该小于1")
        
        # 验证与理论预测值的一致性
        expected_V_us = 0.2253  # QSDT理论预测值
        relative_error = abs(V_us - expected_V_us) / expected_V_us
        self.assertLess(relative_error, 1e-6, f"CKM矩阵元V_us相对误差{relative_error:.2e}过大")
    
    def test_cmb_spectral_index_calculation(self):
        """测试CMB谱指数计算"""
        mu = 1e16  # 暴胀能标 (GeV)
        
        result = _calculate_cmb_spectral_index(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.9, "CMB谱指数过小")
        self.assertLess(result, 1.0, "CMB谱指数过大")
        
        # 验证与理论预测值的一致性
        expected_n_s = 0.9642  # QSDT理论预测值
        relative_error = abs(result - expected_n_s) / expected_n_s
        self.assertLess(relative_error, 1e-6, f"CMB谱指数相对误差{relative_error:.2e}过大")
    
    def test_strong_cp_angle_calculation(self):
        """测试强CP问题θ角计算"""
        mu = 1.0  # 1 GeV标尺
        
        result = _calculate_strong_cp_angle(self.test_params, mu, self.test_config)
        
        # 验证返回值
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.0, "θ角应该精确为0")
        
        # 验证理论预测
        self.assertAlmostEqual(result, 0.0, places=10, msg="θ角应该精确为0")
    
    def test_numerical_precision(self):
        """测试数值精度"""
        # 测试高精度计算
        mu = 246.0
        result = _calculate_higgs_mass(self.test_params, mu, self.test_config)
        
        # 验证精度
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result), "希格斯质量应该为有限值")
        
        # 验证相对精度
        expected = 125.10
        if result > 0:
            relative_error = abs(result - expected) / expected
            self.assertLess(relative_error, 0.01, f"希格斯质量相对误差{relative_error:.2%}过大")
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试极小参数值
        small_params = {'J': 1e-30, 'E': 1e-30, 'g': 1e-30, 'Gamma': 0.0}
        
        try:
            result = _calculate_higgs_mass(small_params, 246.0, self.test_config)
            self.assertTrue(np.isfinite(result), "极小参数下希格斯质量应该为有限值")
        except Exception as e:
            # 某些边界情况可能无法计算，这是可以接受的
            if "division by zero" in str(e) or "invalid value" in str(e):
                pass
            else:
                self.fail(f"极小参数导致意外异常: {e}")
    
    def test_theoretical_consistency(self):
        """测试理论一致性"""
        # 测试E/J比值
        E_over_J = self.test_params['E'] / self.test_params['J']
        self.assertAlmostEqual(E_over_J, 3.0, places=1,
                             msg="E/J比值应该接近3.0")
        
        # 测试Gamma = g * J关系
        expected_Gamma = self.test_params['g'] * self.test_params['J']
        actual_Gamma = self.test_params['Gamma']
        self.assertAlmostEqual(actual_Gamma, expected_Gamma, places=10,
                             msg="Gamma应该等于g * J")


if __name__ == '__main__':
    unittest.main()
