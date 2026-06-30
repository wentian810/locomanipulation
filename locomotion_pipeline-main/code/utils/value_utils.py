import numpy as np
from typing import List, Dict, Deque
from collections import deque

def _normalize_breakpoints(
    breakpoints: List[int],
    values_length: int,
) -> List[int]:
    normalized = sorted(set(breakpoints))

    for breakpoint in normalized:
        if breakpoint < 0 or breakpoint > values_length:
            raise ValueError(
                f"breakpoints must be between 0 and {values_length}, got {breakpoint}"
            )

    return normalized


class SeqValueAnalyzer:
    def __init__(self, ):
        pass

    @staticmethod
    def longest_intervals_by_gap_with_breakpoints(
        values: np.ndarray,
        breakpoints: List[int],
        max_gap: float,
        min_length: int,
        include_breakpoints: bool = True,
    ):
        """
        Split values by breakpoint indices, then query each long enough segment.

        When include_breakpoints is True, each breakpoint index is included in
        the segment after it: [0, bp), [bp, next_bp).
        When include_breakpoints is False, breakpoint indices are skipped:
        [0, bp), [bp + 1, next_bp).
        """
        if min_length < 0:
            raise ValueError(f"min_length must be non-negative, got {min_length}")

        normalized_breakpoints = _normalize_breakpoints(breakpoints, len(values))
        periods: List[Dict[str, object]] = []
        segment_start = 0

        for breakpoint in [*normalized_breakpoints, len(values)]:
            segment_end = breakpoint
            if segment_end - segment_start <= min_length:
                segment_start = breakpoint if include_breakpoints else min(breakpoint + 1, len(values))
                continue

            segment_values = values[segment_start:segment_end]
            interval = SeqValueAnalyzer.longest_interval_by_gap(segment_values, max_gap)

            if interval["length"] > min_length:
                local_start = interval["start"]
                local_end = interval["end"]
                if local_start is not None and local_end is not None:
                    periods.append({
                        "start": segment_start + int(local_start),
                        "end": segment_start + int(local_end),
                        "length": interval["length"],
                    })

            segment_start = breakpoint if include_breakpoints else min(breakpoint + 1, len(values))

        return periods

    @staticmethod
    def longest_interval_by_gap(
        values: np.ndarray,
        max_gap: float,
    ) -> Dict[str, object]:
        """
        挑选最长连续区间，要求区间内 max(value) - min(value) <= max_gap。

        这个规则是区间级别的，不是单个元素级别的。例如：
            [1.0, 1.2, 1.4] 在 max_gap=0.5 时满足规则
            [1.0, 1.2, 1.8] 在 max_gap=0.5 时不满足规则

        返回区间使用 [start, end) 格式。
        """
        if max_gap < 0:
            raise ValueError(f"max_gap must be non-negative, got {max_gap}")

        min_queue: Deque[int] = deque()
        max_queue: Deque[int] = deque()
        left = 0
        best_start = 0
        best_end = 0

        for right, value in enumerate(values):
            while min_queue and values[min_queue[-1]] > value:
                min_queue.pop()
            min_queue.append(right)

            while max_queue and values[max_queue[-1]] < value:
                max_queue.pop()
            max_queue.append(right)

            while min_queue and max_queue and values[max_queue[0]] - values[min_queue[0]] > max_gap:
                if min_queue[0] == left:
                    min_queue.popleft()
                if max_queue[0] == left:
                    max_queue.popleft()
                left += 1

            cur_end = right + 1
            if cur_end - left > best_end - best_start:
                best_start = left
                best_end = cur_end

        return {
            "start": best_start if best_end > best_start else None,
            "end": best_end if best_end > best_start else None,
            "length": best_end - best_start,
        }

if __name__ == "__main__":
    values = np.array([1.0, 1.2, 1.4, 1.8, 2.0, 2.1])
    breakpoints = [3]
    max_gap = 0.5
    min_length = 2

    analyzer = SeqValueAnalyzer()
    periods = analyzer.longest_intervals_by_gap_with_breakpoints(
        values=values,
        breakpoints=breakpoints,
        max_gap=max_gap,
        min_length=min_length,
        include_breakpoints=True,
    )
    print(periods)
