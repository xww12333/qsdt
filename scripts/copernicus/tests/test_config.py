"""
QSDT哥白尼计划配置测试

功能：测试配置文件的加载和验证
作用：确保配置文件格式正确，参数有效
理论文档位置：附录26 - 单位转换与物理常数
"""

import unittest
import yaml
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import load_config


class TestConfig(unittest.TestCase):
    """配置测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config_example.yaml')
        self.config = load_config(self.config_path)
    
    def test_config_loading(self):
        """测试配置文件加载"""
        self.assertIsInstance(self.config, dict)
        self.assertIn('grid', self.config)
        self.assertIn('predictions', self.config)
        self.assertIn('calibration', self.config)
    
    def test_grid_config(self):
        """测试网格配置"""
        grid = self.config['grid']
        self.assertIn('stages', grid)
        self.assertIsInstance(grid['stages'], list)
        # 注意：实际配置文件中没有mu_GeV字段，只有stages
    
    def test_predictions_config(self):
        """测试预测配置"""
        predictions = self.config['predictions']
        self.assertIn('observables', predictions)
        self.assertIn('higgs', predictions)
        self.assertIn('delta_mnp', predictions)
        # 注意：lepton_map在use_lepton_map为True时从外部文件加载
    
    def test_higgs_config(self):
        """测试希格斯质量配置"""
        higgs = self.config['predictions']['higgs']
        self.assertIn('k_mu', higgs)
        self.assertIn('J_to_GeV', higgs)
        self.assertIn('xi_E_over_J', higgs)
        
        # 验证关键参数
        self.assertEqual(higgs['k_mu'], 379212.5)  # 校准值以获得正确的希格斯质量
        # J_to_GeV在配置文件中是字符串，需要转换为浮点数
        J_to_GeV = float(higgs['J_to_GeV'])
        self.assertEqual(J_to_GeV, 6.241509074e9)  # 单位转换因子
        self.assertEqual(higgs['xi_E_over_J'], 3.0)  # E/J比值
    
    def test_delta_mnp_config(self):
        """测试质子-中子质量差配置"""
        delta_mnp = self.config['predictions']['delta_mnp']
        self.assertIn('C_EM', delta_mnp)
        self.assertIn('C_QCD', delta_mnp)
        
        # 验证理论推导的系数
        self.assertEqual(delta_mnp['C_EM'], 0.331)
        self.assertEqual(delta_mnp['C_QCD'], 0.292)
    
    def test_lepton_map_config(self):
        """测试轻子映射配置"""
        # 注意：lepton_map从外部文件加载，不在主配置中
        # 这里测试use_lepton_map标志
        predictions = self.config['predictions']
        self.assertIn('use_lepton_map', predictions)
        self.assertTrue(predictions['use_lepton_map'], "use_lepton_map应该为True")
    
    def test_calibration_config(self):
        """测试校准配置"""
        calibration = self.config['calibration']
        # 注意：calibration可能为None，需要检查
        if calibration is not None:
            self.assertIn('rg', calibration)
            self.assertIn('running_constants', calibration)
    
    def test_observables_list(self):
        """测试可观测量列表"""
        observables = self.config['predictions']['observables']
        expected_observables = [
            'Higgs_mass', 'm_e_MeV', 'm_mu_MeV', 'm_tau_MeV',
            'delta_m_np', 'sin2_theta_W', 'm_t_GeV', 'm_d_mu_diff_MeV',
            'm_u_MeV', 'm_d_MeV', 'm_s_MeV', 'm_c_GeV', 'm_b_GeV',
            'm_W_GeV', 'm_Z_GeV', 'electron_g2', 'a_e', 'V_us',
            'n_s', 'cmb_spectral_index', 'theta_angle', 'strong_cp_angle'
        ]
        
        for obs in expected_observables:
            self.assertIn(obs, observables, f"可观测量 {obs} 缺失")
    
    def test_unit_conversion_factors(self):
        """测试单位转换因子"""
        higgs = self.config['predictions']['higgs']
        J_to_GeV = float(higgs['J_to_GeV'])  # 转换为浮点数
        
        # 验证单位转换因子的物理意义
        # 1 GeV = 1.602176634e-10 J
        GeV_to_J = 1.0 / J_to_GeV
        expected_GeV_to_J = 1.602176634e-10
        
        self.assertAlmostEqual(GeV_to_J, expected_GeV_to_J, places=10,
                             msg="单位转换因子不正确")
    
    def test_theoretical_parameters(self):
        """测试理论参数的一致性"""
        higgs = self.config['predictions']['higgs']
        delta_mnp = self.config['predictions']['delta_mnp']
        
        # 验证k_mu参数
        self.assertEqual(higgs['k_mu'], 379212.5, "k_mu参数与校准值不符")
        
        # 验证C_EM和C_QCD参数
        self.assertEqual(delta_mnp['C_EM'], 0.331, "C_EM参数与理论推导值不符")
        self.assertEqual(delta_mnp['C_QCD'], 0.292, "C_QCD参数与理论推导值不符")


if __name__ == '__main__':
    unittest.main()
