import csv
from typing import Dict, List, Tuple
from pathlib import Path


class SequenceStatistics:
    """
    用来统计多个sequence的属性类
    支持增删改查、计算平均合格率、导出CSV表格
    """
    
    def __init__(self):
        """初始化统计数据存储"""
        self.sequences: Dict[str, Dict[str, int]] = {}
    
    def add(self, sequence_id: str, total: int, qualified: int) -> None:
        """
        增：添加一个sequence的统计数据
        
        Args:
            sequence_id: sequence的唯一标识
            total: 该sequence的总数
            qualified: 该sequence的合格数
            
        Raises:
            ValueError: 当数据非法时抛出异常
        """
        if total < 0 or qualified < 0:
            raise ValueError("总数和合格数必须为非负数")
        if qualified > total:
            raise ValueError("合格数不能超过总数")
        
        self.sequences[sequence_id] = {
            'total': total,
            'qualified': qualified
        }
    
    def delete(self, sequence_id: str) -> None:
        """
        删：删除一个sequence的统计数据
        
        Args:
            sequence_id: 要删除的sequence的唯一标识
            
        Raises:
            ValueError: 当sequence不存在时抛出异常
        """
        if sequence_id not in self.sequences:
            raise ValueError(f"Sequence '{sequence_id}' 不存在")
        
        del self.sequences[sequence_id]
    
    def update(self, sequence_id: str, total: int = None, qualified: int = None) -> None:
        """
        改：更新一个sequence的统计数据
        
        Args:
            sequence_id: 要更新的sequence的唯一标识
            total: 新的总数（可选）
            qualified: 新的合格数（可选）
            
        Raises:
            ValueError: 当sequence不存在或数据非法时抛出异常
        """
        if sequence_id not in self.sequences:
            raise ValueError(f"Sequence '{sequence_id}' 不存在")
        
        new_total = total if total is not None else self.sequences[sequence_id]['total']
        new_qualified = qualified if qualified is not None else self.sequences[sequence_id]['qualified']
        
        if new_total < 0 or new_qualified < 0:
            raise ValueError("总数和合格数必须为非负数")
        if new_qualified > new_total:
            raise ValueError("合格数不能超过总数")
        
        self.sequences[sequence_id]['total'] = new_total
        self.sequences[sequence_id]['qualified'] = new_qualified
    
    def is_empty(self) -> bool:
        """检查是否没有任何sequence数据"""
        return len(self.sequences) == 0

    def get(self, sequence_id: str) -> Dict[str, int]:
        """
        查：查询一个sequence的统计数据
        
        Args:
            sequence_id: 要查询的sequence的唯一标识
            
        Returns:
            包含'total'和'qualified'的字典
            
        Raises:
            ValueError: 当sequence不存在时抛出异常
        """
        if sequence_id not in self.sequences:
            raise ValueError(f"Sequence '{sequence_id}' 不存在")
        
        return self.sequences[sequence_id].copy()
    
    def get_all(self) -> Dict[str, Dict[str, int]]:
        """
        查：获取所有sequences的统计数据
        
        Returns:
            所有sequence的统计数据副本
        """
        return {k: v.copy() for k, v in self.sequences.items()}
    
    def get_qualified_rate(self, sequence_id: str) -> float:
        """
        查：查询单个sequence的合格率
        
        Args:
            sequence_id: sequence的唯一标识
            
        Returns:
            合格率（0-1之间）
            
        Raises:
            ValueError: 当sequence不存在时抛出异常
        """
        if sequence_id not in self.sequences:
            raise ValueError(f"Sequence '{sequence_id}' 不存在")
        
        data = self.sequences[sequence_id]
        if data['total'] == 0:
            return 0.0
        
        return data['qualified'] / data['total']
    
    def get_average_qualified_rate(self) -> float:
        """
        查：计算平均合格率（所有sequence的合格数总和 / 所有sequence的总数总和）
        
        Returns:
            平均合格率（0-1之间），当没有数据时返回0
        """
        if not self.sequences:
            return 0.0
        
        total_qualified = sum(data['qualified'] for data in self.sequences.values())
        total_count = sum(data['total'] for data in self.sequences.values())
        
        if total_count == 0:
            return 0.0
        
        return total_qualified / total_count
    
    def export_csv(self, filepath: str) -> None:
        """
        导出CSV表格
        
        Args:
            filepath: 要保存的CSV文件路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['Sequence ID', 'Total', 'Qualified', 'Qualified Rate'])
            
            # 写入每个sequence的数据
            for seq_id in sorted(self.sequences.keys()):
                data = self.sequences[seq_id]
                rate = data['qualified'] / data['total'] if data['total'] > 0 else 0
                writer.writerow([seq_id, data['total'], data['qualified'], f"{rate:.2%}"])
            
            # 写入总体统计
            total_qualified = sum(data['qualified'] for data in self.sequences.values())
            total_count = sum(data['total'] for data in self.sequences.values())
            avg_rate = total_qualified / total_count if total_count > 0 else 0
            writer.writerow(['总体平均', total_count, total_qualified, f"{avg_rate:.2%}"])
    
    def get_summary(self, filter_ratio_list:List[float]=[1.0]) -> Dict:
        """
        获取统计摘要信息
        
        Returns:
            包含总数、合格数、平均合格率等信息的字典
        """
        total_count = sum(data['total'] for data in self.sequences.values())
        total_qualified = sum(data['qualified'] for data in self.sequences.values())
        avg_rate = total_qualified / total_count if total_count > 0 else 0

        qualified_summary = {}
        for ratio in filter_ratio_list:
            ratio_qualified_list = [
                data['qualified'] for data in self.sequences.values() if data['total'] > 0 and data['qualified'] >= data['total'] * ratio
            ]
            ratio_qualified_count = len(ratio_qualified_list)
            ratio_qualified_frames = sum(ratio_qualified_list)
            ratio_qualified_rate = ratio_qualified_count / len(self.sequences) if self.sequences else 0
            qualified_summary[f'bg_{ratio}'] = {
                'qualified_count': ratio_qualified_count,
                'qualified_frames': ratio_qualified_frames,
                'qualified_rate': ratio_qualified_rate
            }
        
        return {
            'sequence_count': len(self.sequences),
            'total_count': total_count,
            'total_qualified': total_qualified,
            'average_qualified_rate': avg_rate,
            'qualified_summary': qualified_summary, 
        }

    # load from CSV and analysis
    def load_from_csv(self, filepath: str) -> None:
        """
        从CSV文件加载统计数据
        
        Args:
            filepath: CSV文件路径
        """
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Sequence ID'] == '总体平均':
                    continue  # 跳过总体平均行
                sequence_id = row['Sequence ID']
                total = int(row['Total'])
                qualified = int(row['Qualified'])
                self.add(sequence_id, total, qualified)
    
    def merge_from_other(self, other: 'SequenceStatistics') -> None:
        for seq_id, data in other.sequences.items():
            self.add(seq_id, data['total'], data['qualified'])

    def __repr__(self) -> str:
        """返回字符串表示"""
        return f"SequenceStatistics(count={len(self.sequences)}, avg_rate={self.get_average_qualified_rate():.2%})"

def recursive_print(data, prefix=''):
    if isinstance(data, dict):
        for key in data.keys():
            print(f"{prefix}{key}:")
            recursive_print(data[key], prefix+'  ')
    elif isinstance(data, list):
        print(f'{prefix}{data}')
    elif isinstance(data, tuple):
        print(f'{prefix}{data}')
    elif isinstance(data, set):
        print(f'{prefix}{data}')
    elif isinstance(data, str):
        print(f'{prefix}{data}')
    elif isinstance(data, float):
        print(f'{prefix}{data}')
    elif isinstance(data, int):
        print(f'{prefix}{data}')
    else:
        print(f'unknown type: {type(data)}')

if __name__ == "__main__":
    # 使用示例
    stats = SequenceStatistics()
    stats.load_from_csv('output/zitai_n10000_idx000-mv_height/sequence_statistics.csv')
    stats_info = stats.get_summary(filter_ratio_list=[1.0, 0.95, 0.9])
    recursive_print(stats_info)
