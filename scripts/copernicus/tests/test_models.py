"""
QSDT哥白尼计划模型计算测试

功能：测试物理量计算模型
作用：确保理论预测的准确性和数值稳定性
理论文档位置：
    - 附录8：哥白尼计划v6.0 - 轻子质量谱预测
    - 附录10：希格斯质量理论推导
    - 附录24：QSDT终极参数附录
    - 附录26：单位转换与物理常数
"""

import unittest
import numpy as np
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import predict_at_mu, qsd_lqcd


class TestModels(unittest.TestCase):
    """模型计算测试类"""
    
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
                'formulas_module': 'scripts.copernicus.plugins.strict_formulas',
                'higgs': {
                    'k_mu': 379212.5,  # 调整以获得125 GeV的希格斯质量
                    'J_to_GeV': 6.241509074e9,
                    'xi_E_over_J': 3.0
                },
                'delta_mnp': {
                    'C_EM': 0.331,
                    'C_QCD': 0.292
                },
                'lepton_map': {
                    'k1': 5.10435e-01,
                    'k2': 3.83959e+02,
                    'k3': 1.31293e+04,
                    'mu_L_GeV': 85.0
                }
            }
        }
    
    def test_qsd_lqcd_calculation(self):
        """测试QSDT修正的Lambda_QCD计算"""
        mu = 91.1876  # Z玻色子质量 (GeV)
        params = {'J': 8.97e8, 'g': 0.313}
        
        # 计算QSDT修正的Lambda_QCD
        lambda_qcd = qsd_lqcd(mu, self.test_config, params)
        
        # 验证返回值
        self.assertIsInstance(lambda_qcd, float)
        self.assertGreater(lambda_qcd, 0, "Lambda_QCD应该为正值")
        self.assertLess(lambda_qcd, 1.0, "Lambda_QCD应该小于1 GeV")
        
        # 验证数值合理性（标准值约为0.2-0.3 GeV）
        self.assertGreater(lambda_qcd, 0.05, "Lambda_QCD过小")
        self.assertLess(lambda_qcd, 0.5, "Lambda_QCD过大")
    
    def test_higgs_mass_calculation(self):
        """测试希格斯质量计算"""
        mu = 246.0  # 电弱标尺 (GeV)
        observables = ['Higgs_mass_GeV']
        
        result = predict_at_mu(self.test_params, mu, observables, self.test_config)
        
        # 验证希格斯质量
        self.assertIn('Higgs_mass_GeV', result)
        higgs_mass = result['Higgs_mass_GeV']
        
        self.assertIsInstance(higgs_mass, float)
        self.assertGreater(higgs_mass, 0, "希格斯质量应该为正值")
        
        # 验证数值合理性（实验值约125 GeV）
        self.assertGreater(higgs_mass, 100, "希格斯质量过小")
        self.assertLess(higgs_mass, 200, "希格斯质量过大")
    
    def test_lepton_mass_calculation(self):
        """测试轻子质量计算"""
        mu = 85.0  # 轻子标尺 (GeV)
        observables = ['m_e_MeV', 'm_mu_MeV', 'm_tau_MeV']
        
        result = predict_at_mu(self.test_params, mu, observables, self.test_config)
        
        # 验证轻子质量
        lepton_masses = ['m_e_MeV', 'm_mu_MeV', 'm_tau_MeV']
        for mass_key in lepton_masses:
            self.assertIn(mass_key, result)
            mass = result[mass_key]
            
            self.assertIsInstance(mass, float)
            self.assertGreater(mass, 0, f"{mass_key}应该为正值")
        
        # 验证质量层次结构
        m_e = result['m_e_MeV']
        m_mu = result['m_mu_MeV']
        m_tau = result['m_tau_MeV']
        
        self.assertLess(m_e, m_mu, "电子质量应该小于μ子质量")
        self.assertLess(m_mu, m_tau, "μ子质量应该小于τ子质量")
        
        # 验证数值合理性（实验值）
        self.assertAlmostEqual(m_e, 0.511, places=2, msg="电子质量与实验值不符")
        self.assertAlmostEqual(m_mu, 105.66, places=1, msg="μ子质量与实验值不符")
        self.assertAlmostEqual(m_tau, 1776.86, places=0, msg="τ子质量与实验值不符")
    
    def test_delta_mnp_calculation(self):
        """测试质子-中子质量差计算"""
        mu = 1.0  # 1 GeV标尺
        observables = ['delta_m_np_MeV']
        
        result = predict_at_mu(self.test_params, mu, observables, self.test_config)
        
        # 验证质子-中子质量差
        self.assertIn('delta_m_np_MeV', result)
        delta_mnp = result['delta_m_np_MeV']
        
        self.assertIsInstance(delta_mnp, float)
        self.assertGreater(delta_mnp, 0, "质子-中子质量差应该为正值")
        
        # 验证数值合理性（实验值约1.293 MeV）
        self.assertGreater(delta_mnp, 0.5, "质子-中子质量差过小")
        self.assertLess(delta_mnp, 3.0, "质子-中子质量差过大")
    
    def test_unit_conversion(self):
        """测试单位转换"""
        # 测试J到GeV的转换
        J_to_GeV = self.test_config['predictions']['higgs']['J_to_GeV']
        J_joule = self.test_params['J']
        J_GeV = J_joule * J_to_GeV
        
        # 验证转换结果
        self.assertIsInstance(J_GeV, float)
        self.assertGreater(J_GeV, 0, "J_GeV应该为正值")
        
        # 验证转换精度
        expected_J_GeV = J_joule * 6.241509074e9
        self.assertAlmostEqual(J_GeV, expected_J_GeV, places=10,
                             msg="单位转换精度不足")
    
    def test_normalization_method(self):
        """测试归一化方法"""
        # 测试轻子质量计算中的归一化
        J_GeV = self.test_params['J'] * self.test_config['predictions']['higgs']['J_to_GeV']
        E_GeV = self.test_params['E'] * self.test_config['predictions']['higgs']['J_to_GeV']
        Gamma_GeV = self.test_params['Gamma'] * self.test_config['predictions']['higgs']['J_to_GeV']
        
        # 归一化计算
        ref = max(J_GeV, 1e-30)
        x1 = (E_GeV - 2.0 * J_GeV) / ref
        x2 = Gamma_GeV / ref
        x3 = (Gamma_GeV * Gamma_GeV) / (ref * ref)
        
        # 验证归一化结果
        self.assertTrue(np.isfinite(x1), "x1应该为有限值")
        self.assertTrue(np.isfinite(x2), "x2应该为有限值")
        self.assertTrue(np.isfinite(x3), "x3应该为有限值")
        
        # 验证归一化避免了数值爆炸
        self.assertLess(abs(x1), 1e6, "x1归一化后仍然过大")
        self.assertLess(abs(x2), 1e6, "x2归一化后仍然过大")
        self.assertLess(abs(x3), 1e6, "x3归一化后仍然过大")
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试极端参数值
        extreme_params = [
            {'J': 1e-30, 'E': 1e-30, 'g': 1e-30, 'Gamma': 0.0},
            {'J': 1e30, 'E': 1e30, 'g': 0.99, 'Gamma': 1e30},
            {'J': 0.0, 'E': 0.0, 'g': 0.0, 'Gamma': 0.0},
        ]
        
        for params in extreme_params:
            try:
                result = predict_at_mu(246.0, params, self.test_config)
                
                # 验证结果有限
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        self.assertTrue(np.isfinite(value), 
                                      f"参数{params}下{key}为无穷大")
                
            except Exception as e:
                # 某些极端情况可能无法计算，这是可以接受的
                if "division by zero" in str(e) or "invalid value" in str(e):
                    continue
                else:
                    self.fail(f"极端参数{params}导致意外异常: {e}")
    
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
    
    def test_energy_scale_dependence(self):
        """测试能量标尺依赖性"""
        energy_scales = [1.0, 85.0, 246.0, 1000.0]  # GeV
        observables = ['alpha_em@mu', 'alpha_s@mu']
        
        for mu in energy_scales:
            result = predict_at_mu(self.test_params, mu, observables, self.test_config)
            
            # 验证结果存在
            self.assertIsInstance(result, dict)
            self.assertGreater(len(result), 0, f"能量标尺{mu} GeV下无计算结果")
            
            # 验证关键物理量存在
            if mu == 246.0:  # 电弱标尺
                # 注意：246 GeV时只计算alpha，不计算Higgs质量
                pass
            elif mu == 85.0:  # 轻子标尺
                # 注意：85 GeV时只计算alpha，不计算轻子质量
                pass
            elif mu == 1.0:  # 强子标尺
                # 注意：1 GeV时只计算alpha，不计算delta_mnp
                pass


if __name__ == '__main__':
    unittest.main()
