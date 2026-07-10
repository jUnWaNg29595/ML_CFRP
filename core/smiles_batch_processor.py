# -*- coding: utf-8 -*-
"""
SMILES 批量处理模块 - 优化版

解决问题：
1. 使用真正的批量GPU推理，而非逐条处理
2. 集成任务管理器，支持进度监控和任务取消
3. 提供详细的进度反馈

Author: ML_CFRP System (Optimized)
"""

from __future__ import annotations

import time
import logging
from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import threading

import numpy as np
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入任务管理器
try:
    from .task_manager import (
        get_task_manager, 
        is_cancelled, 
        clear_cancel,
        TaskStatus
    )
    TASK_MANAGER_AVAILABLE = True
except ImportError:
    TASK_MANAGER_AVAILABLE = False
    def is_cancelled():
        return False
    def clear_cancel():
        pass

# 尝试导入Transformer纠错器
try:
    from .smiles_transformer_corrector import (
        SMILESCorrectionPipeline,
        CorrectionResult,
        TORCH_AVAILABLE,
        RDKIT_AVAILABLE
    )
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    TORCH_AVAILABLE = False
    RDKIT_AVAILABLE = False

# 导入规则修复函数
try:
    from .smiles_utils import (
        canonicalize_smiles,
        smart_repair_smiles,
        aggressive_repair_smiles,
        ultra_repair_smiles,
        clean_smiles_raw_string
    )
    SMILES_UTILS_AVAILABLE = True
except ImportError:
    SMILES_UTILS_AVAILABLE = False
    canonicalize_smiles = None
    smart_repair_smiles = None
    aggressive_repair_smiles = None
    ultra_repair_smiles = None
    clean_smiles_raw_string = lambda x: x


@dataclass
class BatchProcessingConfig:
    """批量处理配置"""
    # GPU推理设置
    batch_size: int = 256           # GPU推理批次大小
    use_fp16: bool = True           # 使用FP16半精度
    use_greedy_first: bool = True   # 优先使用贪婪解码
    beam_size: int = 5              # Beam search大小
    
    # 任务管理
    enable_cancellation: bool = True  # 启用取消检测
    progress_interval: int = 100      # 进度更新间隔（样本数）
    
    # 策略设置
    use_transformer: bool = True      # 使用Transformer纠错
    use_rules_fallback: bool = True   # DL失败后使用规则方法
    preserve_original_on_fail: bool = False  # 失败时保留原始


@dataclass  
class BatchProcessingResult:
    """批量处理结果"""
    corrected_smiles: List[Optional[str]]
    methods: List[str]  # 每个样本使用的方法
    stats: Dict[str, int]
    processing_time: float
    samples_per_second: float


class SMILESBatchProcessor:
    """
    SMILES 批量处理器 - 优化版
    
    特性：
    1. 真正的批量GPU推理，最大化GPU利用率
    2. 集成任务管理器，支持进度监控和取消
    3. 智能批次管理，自动调整batch_size
    4. 详细的统计信息和进度回调
    """
    
    def __init__(self, 
                 config: BatchProcessingConfig = None,
                 model_path: Optional[str] = None):
        """
        初始化批量处理器
        
        Args:
            config: 批量处理配置
            model_path: Transformer模型路径
        """
        self.config = config or BatchProcessingConfig()
        self.model_path = model_path
        
        # 延迟初始化Pipeline
        self._pipeline: Optional[SMILESCorrectionPipeline] = None
        self._pipeline_lock = threading.Lock()
        
        # 任务管理
        self._task_id: Optional[str] = None
        self._task_manager = get_task_manager() if TASK_MANAGER_AVAILABLE else None
        
        # 统计信息
        self.stats = {
            'total_processed': 0,
            'direct_valid': 0,
            'dl_corrected': 0,
            'rule_corrected': 0,
            'failed': 0,
            'cancelled': 0
        }
    
    def _get_pipeline(self) -> Optional[SMILESCorrectionPipeline]:
        """获取或创建Pipeline（线程安全）"""
        if not TRANSFORMER_AVAILABLE:
            return None
            
        with self._pipeline_lock:
            if self._pipeline is None:
                try:
                    self._pipeline = SMILESCorrectionPipeline(
                        use_fp16=self.config.use_fp16
                    )
                    
                    # 加载模型
                    if self.model_path:
                        self._pipeline.load_model(self.model_path)
                    
                    # 自动优化batch_size
                    if TORCH_AVAILABLE and self._pipeline.device.type == 'cuda':
                        self._pipeline.auto_batch_size()
                        
                except Exception as e:
                    logger.error(f"初始化Pipeline失败: {e}")
                    self._pipeline = None
                    
        return self._pipeline
    
    def _register_task(self, name: str, total_items: int):
        """注册任务到任务管理器"""
        if self._task_manager is not None:
            self._task_id = self._task_manager.register_task(
                name=name,
                task_type='transformer_inference',
                total_items=total_items
            )
            self._task_manager.start_task(self._task_id)
            clear_cancel()  # 清除之前的取消标志
    
    def _update_task_progress(self, processed: int):
        """更新任务进度"""
        if self._task_manager is not None and self._task_id:
            self._task_manager.update_progress(self._task_id, processed)
    
    def _complete_task(self, success: bool = True, error_message: str = ""):
        """完成任务"""
        if self._task_manager is not None and self._task_id:
            self._task_manager.complete_task(self._task_id, success, error_message)
            self._task_id = None
    
    def _check_cancelled(self) -> bool:
        """检查是否被取消"""
        if self.config.enable_cancellation and is_cancelled():
            return True
        return False
    
    def process_batch(self,
                     smiles_list: List[str],
                     progress_callback: Callable[[int, int, str, Dict], None] = None,
                     show_progress: bool = True) -> BatchProcessingResult:
        """
        批量处理SMILES - 核心优化方法
        
        使用真正的批量GPU推理，而非逐条处理。
        
        Args:
            smiles_list: SMILES列表
            progress_callback: 进度回调函数 (current, total, stage, info)
            show_progress: 是否显示进度条
            
        Returns:
            BatchProcessingResult
        """
        if not smiles_list:
            return BatchProcessingResult(
                corrected_smiles=[],
                methods=[],
                stats=self.stats.copy(),
                processing_time=0.0,
                samples_per_second=0.0
            )
        
        start_time = time.time()
        total_samples = len(smiles_list)
        
        # 注册任务
        self._register_task(f"SMILES批量纠错 ({total_samples}条)", total_samples)
        
        # 初始化结果
        results = [None] * total_samples
        methods = ['unknown'] * total_samples
        
        # 重置统计
        local_stats = {
            'total_processed': 0,
            'direct_valid': 0,
            'dl_corrected': 0,
            'rule_corrected': 0,
            'failed': 0,
            'cancelled': 0
        }
        
        try:
            # ========== 阶段1：预处理和快速验证 ==========
            if show_progress:
                print(f"\n{'='*60}")
                print(f"🚀 SMILES批量处理 - 共 {total_samples:,} 条")
                print(f"{'='*60}")
                print(f"\n[阶段 1/3] 预处理和快速验证...")
            
            preprocessed = []
            needs_correction_indices = []
            
            for i, smiles in enumerate(smiles_list):
                # 检查取消
                if self._check_cancelled():
                    local_stats['cancelled'] = total_samples - i
                    self._complete_task(False, "用户取消")
                    break
                
                # 预处理
                cleaned = clean_smiles_raw_string(smiles) if SMILES_UTILS_AVAILABLE else smiles
                
                if not cleaned:
                    results[i] = None
                    methods[i] = 'invalid_input'
                    local_stats['failed'] += 1
                elif canonicalize_smiles and canonicalize_smiles(cleaned):
                    # 直接有效
                    results[i] = canonicalize_smiles(cleaned)
                    methods[i] = 'direct'
                    local_stats['direct_valid'] += 1
                else:
                    # 需要纠错
                    preprocessed.append(cleaned)
                    needs_correction_indices.append(i)
                
                local_stats['total_processed'] += 1
                
                # 更新进度
                if i % self.config.progress_interval == 0:
                    self._update_task_progress(i + 1)
                    if progress_callback:
                        progress_callback(i + 1, total_samples, '预处理', local_stats)
            
            if show_progress:
                print(f"   ✅ 直接有效: {local_stats['direct_valid']:,} 条")
                print(f"   🔧 需要纠错: {len(preprocessed):,} 条")
            
            # ========== 阶段2：批量Transformer纠错 ==========
            if self.config.use_transformer and preprocessed and not self._check_cancelled():
                pipeline = self._get_pipeline()
                
                if pipeline is not None:
                    if show_progress:
                        print(f"\n[阶段 2/3] 批量Transformer纠错...")
                        print(f"   📋 配置: batch_size={self.config.batch_size}, FP16={self.config.use_fp16}")
                    
                    # ★ 真正的批量推理！
                    batch_results = pipeline.correct_batch(
                        preprocessed,
                        use_dl=True,
                        use_rules=False,  # 先只用DL
                        beam_size=self.config.beam_size,
                        show_progress=show_progress,
                        batch_size=self.config.batch_size,
                        progress_callback=lambda cur, tot, stage, info: (
                            self._update_task_progress(local_stats['direct_valid'] + cur),
                            progress_callback(local_stats['direct_valid'] + cur, total_samples, f'DL纠错-{stage}', info) if progress_callback else None
                        )
                    )
                    
                    # 处理DL结果
                    still_needs_rule = []
                    still_needs_rule_indices = []
                    
                    for i, (idx, result) in enumerate(zip(needs_correction_indices, batch_results)):
                        if result:
                            results[idx] = result
                            methods[idx] = 'transformer'
                            local_stats['dl_corrected'] += 1
                        else:
                            still_needs_rule.append(preprocessed[i])
                            still_needs_rule_indices.append(idx)
                    
                    if show_progress:
                        print(f"   ✅ DL纠错成功: {local_stats['dl_corrected']:,} 条")
                        print(f"   ⏭️  需要规则修复: {len(still_needs_rule):,} 条")
                else:
                    still_needs_rule = preprocessed
                    still_needs_rule_indices = needs_correction_indices
            else:
                still_needs_rule = preprocessed
                still_needs_rule_indices = needs_correction_indices
            
            # ========== 阶段3：规则修复 ==========
            if self.config.use_rules_fallback and still_needs_rule and not self._check_cancelled():
                if show_progress:
                    print(f"\n[阶段 3/3] 规则修复...")
                
                for i, (idx, cleaned) in enumerate(zip(still_needs_rule_indices, still_needs_rule)):
                    # 检查取消
                    if self._check_cancelled():
                        local_stats['cancelled'] += len(still_needs_rule) - i
                        break
                    
                    # 尝试规则修复
                    rule_result = None
                    rule_method = None
                    
                    # 1. smart_repair
                    if smart_repair_smiles:
                        try:
                            repaired = smart_repair_smiles(cleaned, keep_largest_frag=True)
                            if repaired and canonicalize_smiles(repaired):
                                rule_result = canonicalize_smiles(repaired)
                                rule_method = 'smart_repair'
                        except:
                            pass
                    
                    # 2. aggressive_repair
                    if rule_result is None and aggressive_repair_smiles:
                        try:
                            repaired = aggressive_repair_smiles(cleaned, keep_largest_frag=True)
                            if repaired and canonicalize_smiles(repaired):
                                rule_result = canonicalize_smiles(repaired)
                                rule_method = 'aggressive_repair'
                        except:
                            pass
                    
                    # 3. ultra_repair
                    if rule_result is None and ultra_repair_smiles:
                        try:
                            repaired, status = ultra_repair_smiles(cleaned, keep_largest_frag=True)
                            if repaired and canonicalize_smiles(repaired):
                                rule_result = canonicalize_smiles(repaired)
                                rule_method = f'ultra_repair_{status}'
                        except:
                            pass
                    
                    if rule_result:
                        results[idx] = rule_result
                        methods[idx] = rule_method
                        local_stats['rule_corrected'] += 1
                    else:
                        # 保留原始或失败
                        if self.config.preserve_original_on_fail:
                            results[idx] = smiles_list[idx]
                            methods[idx] = 'preserved_original'
                        else:
                            results[idx] = None
                            methods[idx] = 'failed'
                            local_stats['failed'] += 1
                    
                    # 更新进度
                    if i % self.config.progress_interval == 0:
                        processed = local_stats['direct_valid'] + local_stats['dl_corrected'] + i + 1
                        self._update_task_progress(processed)
                        if progress_callback:
                            progress_callback(processed, total_samples, '规则修复', local_stats)
                
                if show_progress:
                    print(f"   ✅ 规则修复成功: {local_stats['rule_corrected']:,} 条")
            
            # ========== 完成 ==========
            processing_time = time.time() - start_time
            samples_per_second = total_samples / processing_time if processing_time > 0 else 0
            
            if show_progress:
                success_count = local_stats['direct_valid'] + local_stats['dl_corrected'] + local_stats['rule_corrected']
                success_rate = success_count / total_samples * 100 if total_samples > 0 else 0
                
                print(f"\n{'='*60}")
                print(f"✅ 处理完成!")
                print(f"{'='*60}")
                print(f"📊 统计汇总:")
                print(f"   总计处理: {total_samples:,} 条")
                print(f"   成功率: {success_rate:.1f}%")
                print(f"   - 直接有效: {local_stats['direct_valid']:,}")
                print(f"   - DL纠错: {local_stats['dl_corrected']:,}")
                print(f"   - 规则修复: {local_stats['rule_corrected']:,}")
                print(f"   - 失败: {local_stats['failed']:,}")
                if local_stats['cancelled'] > 0:
                    print(f"   - 取消: {local_stats['cancelled']:,}")
                print(f"⏱️  总耗时: {processing_time:.2f}秒")
                print(f"🚀 处理速度: {samples_per_second:.1f} 条/秒")
                print(f"{'='*60}")
            
            # 更新全局统计
            for key in self.stats:
                self.stats[key] += local_stats.get(key, 0)
            
            # 完成任务
            self._complete_task(True)
            
            return BatchProcessingResult(
                corrected_smiles=results,
                methods=methods,
                stats=local_stats,
                processing_time=processing_time,
                samples_per_second=samples_per_second
            )
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            self._complete_task(False, str(e))
            raise
    
    def process_dataframe_column(self,
                                  df: pd.DataFrame,
                                  column: str,
                                  progress_callback: Callable = None,
                                  show_progress: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理DataFrame中的SMILES列
        
        Args:
            df: 输入DataFrame
            column: SMILES列名
            progress_callback: 进度回调
            show_progress: 是否显示进度
            
        Returns:
            (处理后的DataFrame, 统计信息)
        """
        if column not in df.columns:
            raise ValueError(f"列 '{column}' 不存在")
        
        # 获取SMILES列表
        smiles_list = df[column].fillna('').astype(str).tolist()
        
        # 批量处理
        result = self.process_batch(
            smiles_list,
            progress_callback=progress_callback,
            show_progress=show_progress
        )
        
        # 更新DataFrame
        df_result = df.copy()
        df_result[column] = result.corrected_smiles
        
        # 可选：添加方法列
        df_result[f'{column}_method'] = result.methods
        
        return df_result, {
            'stats': result.stats,
            'processing_time': result.processing_time,
            'samples_per_second': result.samples_per_second
        }
    
    def diagnose(self):
        """诊断处理器状态"""
        print("=" * 60)
        print("🔍 SMILES批量处理器诊断")
        print("=" * 60)
        
        print(f"\n📋 配置:")
        print(f"   batch_size: {self.config.batch_size}")
        print(f"   use_fp16: {self.config.use_fp16}")
        print(f"   use_transformer: {self.config.use_transformer}")
        print(f"   use_rules_fallback: {self.config.use_rules_fallback}")
        
        print(f"\n📦 依赖状态:")
        print(f"   TRANSFORMER_AVAILABLE: {TRANSFORMER_AVAILABLE}")
        print(f"   TORCH_AVAILABLE: {TORCH_AVAILABLE}")
        print(f"   RDKIT_AVAILABLE: {RDKIT_AVAILABLE}")
        print(f"   SMILES_UTILS_AVAILABLE: {SMILES_UTILS_AVAILABLE}")
        print(f"   TASK_MANAGER_AVAILABLE: {TASK_MANAGER_AVAILABLE}")
        
        if TRANSFORMER_AVAILABLE:
            pipeline = self._get_pipeline()
            if pipeline:
                print(f"\n🤖 Pipeline状态:")
                print(f"   设备: {pipeline.device}")
                print(f"   模型加载: {pipeline.model is not None}")
                print(f"   FP16: {pipeline.use_fp16}")
                
                if TORCH_AVAILABLE:
                    import torch
                    if torch.cuda.is_available():
                        print(f"\n🎮 GPU状态:")
                        print(f"   GPU: {torch.cuda.get_device_name(0)}")
                        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                        print(f"   已使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        print("=" * 60)


# =============================================================================
# 便捷函数
# =============================================================================

def batch_correct_smiles(smiles_list: List[str],
                        use_transformer: bool = True,
                        use_rules: bool = True,
                        batch_size: int = 256,
                        use_fp16: bool = True,
                        model_path: str = None,
                        show_progress: bool = True) -> List[Optional[str]]:
    """
    便捷函数：批量修复SMILES
    
    这是推荐的高性能批量处理接口。
    
    Args:
        smiles_list: SMILES列表
        use_transformer: 使用Transformer纠错
        use_rules: DL失败后使用规则方法
        batch_size: GPU推理批次大小
        use_fp16: 使用FP16半精度
        model_path: 模型路径
        show_progress: 显示进度
        
    Returns:
        修复后的SMILES列表
    """
    config = BatchProcessingConfig(
        batch_size=batch_size,
        use_fp16=use_fp16,
        use_transformer=use_transformer,
        use_rules_fallback=use_rules
    )
    
    processor = SMILESBatchProcessor(config=config, model_path=model_path)
    result = processor.process_batch(smiles_list, show_progress=show_progress)
    
    return result.corrected_smiles


def get_batch_processor(config: BatchProcessingConfig = None,
                       model_path: str = None) -> SMILESBatchProcessor:
    """获取批量处理器实例"""
    return SMILESBatchProcessor(config=config, model_path=model_path)


# =============================================================================
# 测试
# =============================================================================

if __name__ == '__main__':
    # 测试批量处理
    test_smiles = [
        'CCO',
        'CC(=O)OC1=CC=CC=C1C(=O)O',
        'CC(=O)OC1=CC=CC=C1C(=O)O)',  # 错误的括号
        'c1ccccc',  # 不完整的环
        'invalid_smiles',
        '[nH]1cccc1',
        'CCN(CC)CC',
    ] * 100  # 扩展到700个样本
    
    processor = SMILESBatchProcessor()
    processor.diagnose()
    
    result = processor.process_batch(test_smiles, show_progress=True)
    
    print(f"\n处理结果示例:")
    for i in range(min(7, len(result.corrected_smiles))):
        print(f"  {test_smiles[i]} -> {result.corrected_smiles[i]} ({result.methods[i]})")
