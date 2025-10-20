#!/usr/bin/env python3
"""
QSDT哥白尼计划测试运行器

功能：运行所有单元测试并生成测试报告
作用：确保QSDT理论计算的准确性和可靠性
理论文档位置：附录26 - 单位转换与物理常数
"""

import unittest
import sys
import os
import time
import argparse
from io import StringIO

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入所有测试模块
from tests.test_config import TestConfig
from tests.test_rg import TestRG
from tests.test_models import TestModels
from tests.test_strict_formulas import TestStrictFormulas
from tests.test_pipeline import TestPipeline


def run_all_tests(verbose=False, failfast=False):
    """运行所有测试"""
    print("=" * 80)
    print("QSDT哥白尼计划单元测试")
    print("=" * 80)
    print()
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestConfig,
        TestRG,
        TestModels,
        TestStrictFormulas,
        TestPipeline
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2 if verbose else 1,
        failfast=failfast
    )
    
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # 输出结果
    print(stream.getvalue())
    
    # 统计信息
    print("=" * 80)
    print("测试统计")
    print("=" * 80)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"执行时间: {end_time - start_time:.2f}秒")
    print()
    
    # 失败详情
    if result.failures:
        print("失败详情:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"❌ {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
        print()
    
    # 错误详情
    if result.errors:
        print("错误详情:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"💥 {test}: {traceback.split('\\n')[-2]}")
        print()
    
    # 成功率
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 所有测试通过！")
    elif success_rate >= 90:
        print("✅ 测试基本通过，有少量问题需要修复")
    elif success_rate >= 70:
        print("⚠️ 测试部分通过，需要修复一些问题")
    else:
        print("❌ 测试失败较多，需要大量修复")
    
    return result.wasSuccessful()


def run_specific_test(test_name, verbose=False):
    """运行特定测试"""
    print(f"运行测试: {test_name}")
    print("=" * 50)
    
    # 根据测试名称选择测试类
    test_classes = {
        'config': TestConfig,
        'rg': TestRG,
        'models': TestModels,
        'formulas': TestStrictFormulas,
        'pipeline': TestPipeline
    }
    
    if test_name not in test_classes:
        print(f"错误: 未知的测试名称 '{test_name}'")
        print(f"可用的测试: {', '.join(test_classes.keys())}")
        return False
    
    # 运行特定测试
    suite = unittest.TestLoader().loadTestsFromTestCase(test_classes[test_name])
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='QSDT哥白尼计划测试运行器')
    parser.add_argument('--test', '-t', help='运行特定测试 (config, rg, models, formulas, pipeline)')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--failfast', '-f', action='store_true', help='遇到第一个失败就停止')
    
    args = parser.parse_args()
    
    if args.test:
        success = run_specific_test(args.test, args.verbose)
    else:
        success = run_all_tests(args.verbose, args.failfast)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
