"""
数据泄漏检查模块
================

检查跨负载场景中是否存在数据泄漏。
"""

from typing import Any, Dict, List, Set, Tuple
import numpy as np


class DataLeakageChecker:
    """
    数据泄漏检查器
    
    检查 source 和 target 之间是否存在数据泄漏。
    """
    
    def __init__(self):
        self.checks_passed = []
        self.checks_failed = []
        self.warnings = []
        
    def check_file_disjoint(self, source_files: Set[str], target_files: Set[str]) -> bool:
        """
        检查 source 和 target 文件集合是否不相交
        
        Args:
            source_files: 源域文件集合
            target_files: 目标域文件集合
            
        Returns:
            是否通过检查
        """
        intersection = source_files & target_files
        if intersection:
            self.checks_failed.append({
                'check': 'file_disjoint',
                'status': 'FAIL',
                'detail': f'Source 和 target 存在共同文件: {intersection}',
            })
            return False
        else:
            self.checks_passed.append({
                'check': 'file_disjoint',
                'status': 'PASS',
                'detail': f'Source files: {source_files}, Target files: {target_files}',
            })
            return True
            
    def check_load_different(self, source_load: int, target_load: int) -> bool:
        """
        检查 source 和 target 负载是否不同
        
        Args:
            source_load: 源域负载
            target_load: 目标域负载
            
        Returns:
            是否通过检查
        """
        if source_load == target_load:
            self.checks_failed.append({
                'check': 'load_different',
                'status': 'FAIL',
                'detail': f'Source load ({source_load}) == Target load ({target_load})',
            })
            return False
        else:
            self.checks_passed.append({
                'check': 'load_different',
                'status': 'PASS',
                'detail': f'Source load: {source_load}, Target load: {target_load}',
            })
            return True
            
    def check_provenance_unique(self, source_provenance: List[Dict], target_provenance: List[Dict]) -> bool:
        """
        检查 provenance 是否唯一（无重复）
        
        Args:
            source_provenance: 源域 provenance 列表
            target_provenance: 目标域 provenance 列表
            
        Returns:
            是否通过检查
        """
        # 检查 source 内部无重复
        source_keys = set()
        source_dupes = 0
        for p in source_provenance:
            key = (p['source_file'], p['window_start'], p['window_end'])
            if key in source_keys:
                source_dupes += 1
            source_keys.add(key)
            
        # 检查 target 内部无重复
        target_keys = set()
        target_dupes = 0
        for p in target_provenance:
            key = (p['source_file'], p['window_start'], p['window_end'])
            if key in target_keys:
                target_dupes += 1
            target_keys.add(key)
            
        # 检查 source 和 target 之间无重复
        cross_dupes = source_keys & target_keys
        
        all_passed = True
        
        if source_dupes > 0:
            self.checks_failed.append({
                'check': 'provenance_unique_source',
                'status': 'FAIL',
                'detail': f'Source 内部存在 {source_dupes} 个重复 provenance',
            })
            all_passed = False
        else:
            self.checks_passed.append({
                'check': 'provenance_unique_source',
                'status': 'PASS',
                'detail': f'Source provenance 无重复 ({len(source_provenance)} 条)',
            })
            
        if target_dupes > 0:
            self.checks_failed.append({
                'check': 'provenance_unique_target',
                'status': 'FAIL',
                'detail': f'Target 内部存在 {target_dupes} 个重复 provenance',
            })
            all_passed = False
        else:
            self.checks_passed.append({
                'check': 'provenance_unique_target',
                'status': 'PASS',
                'detail': f'Target provenance 无重复 ({len(target_provenance)} 条)',
            })
            
        if cross_dupes:
            self.checks_failed.append({
                'check': 'provenance_unique_cross',
                'status': 'FAIL',
                'detail': f'Source 和 target 之间存在 {len(cross_dupes)} 个重复 provenance',
            })
            all_passed = False
        else:
            self.checks_passed.append({
                'check': 'provenance_unique_cross',
                'status': 'PASS',
                'detail': 'Source 和 target provenance 无交叉',
            })
            
        return all_passed
        
    def check_window_overlap(self, source_provenance: List[Dict], target_provenance: List[Dict]) -> bool:
        """
        检查 source 和 target 窗口是否来自同一原始信号且存在重叠
        
        注意：由于 source 和 target 来自不同负载（不同 .mat 文件），
        窗口重叠检查实际上是检查是否来自同一文件。
        
        Args:
            source_provenance: 源域 provenance 列表
            target_provenance: 目标域 provenance 列表
            
        Returns:
            是否通过检查
        """
        # 由于 source 和 target 来自不同负载（不同 .mat 文件），
        # 只要文件不相同，就不存在窗口重叠
        source_files = set(p['source_file'] for p in source_provenance)
        target_files = set(p['source_file'] for p in target_provenance)
        
        return self.check_file_disjoint(source_files, target_files)
        
    def check_adabn_label_isolation(self, adabn_uses_labels: bool) -> bool:
        """
        检查 AdaBN 是否使用了 target labels
        
        Args:
            adabn_uses_labels: AdaBN 是否使用 labels
            
        Returns:
            是否通过检查
        """
        if adabn_uses_labels:
            self.checks_failed.append({
                'check': 'adabn_label_isolation',
                'status': 'FAIL',
                'detail': 'AdaBN 使用了 target labels（违反论文 Algorithm 1）',
            })
            return False
        else:
            self.checks_passed.append({
                'check': 'adabn_label_isolation',
                'status': 'PASS',
                'detail': 'AdaBN 未使用 target labels',
            })
            return True
            
    def run_all_checks(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行所有泄漏检查
        
        Args:
            scenario_data: create_scenario() 返回的数据
            
        Returns:
            检查结果字典
        """
        source = scenario_data['source']
        target = scenario_data['target']
        
        # 运行检查
        self.check_file_disjoint(source['files'], target['files'])
        self.check_load_different(source['load'], target['load'])
        self.check_provenance_unique(source['provenance'], target['provenance'])
        self.check_window_overlap(source['provenance'], target['provenance'])
        
        # 汇总结果
        total_checks = len(self.checks_passed) + len(self.checks_failed)
        passed = len(self.checks_failed) == 0
        
        return {
            'passed': passed,
            'total_checks': total_checks,
            'passed_checks': len(self.checks_passed),
            'failed_checks': len(self.checks_failed),
            'details': {
                'passed': self.checks_passed,
                'failed': self.checks_failed,
                'warnings': self.warnings,
            }
        }
        
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        生成检查报告
        
        Args:
            results: run_all_checks() 返回的结果
            
        Returns:
            Markdown 格式的报告
        """
        report = []
        report.append("# 数据泄漏检查报告\n")
        
        status = "✅ PASS" if results['passed'] else "❌ FAIL"
        report.append(f"## 总体状态: {status}\n")
        report.append(f"- 总检查数: {results['total_checks']}")
        report.append(f"- 通过: {results['passed_checks']}")
        report.append(f"- 失败: {results['failed_checks']}\n")
        
        if results['details']['passed']:
            report.append("## ✅ 通过的检查\n")
            for check in results['details']['passed']:
                report.append(f"- **{check['check']}**: {check['detail']}")
                
        if results['details']['failed']:
            report.append("\n## ❌ 失败的检查\n")
            for check in results['details']['failed']:
                report.append(f"- **{check['check']}**: {check['detail']}")
                
        return "\n".join(report)
