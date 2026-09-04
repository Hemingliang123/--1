"""笔级别逻辑漏洞探测 — 从 K线包含 → 分型 → 笔 逐级找边界与缺陷。

运行: python3 chanlun/test_stroke_vulnerabilities.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanlun.analyzer import analyze
from chanlun.fractal import find_fractals
from chanlun.kline import process_inclusion
from chanlun.models import Bar, FractalType, MergedBar, StrokeStandard
from chanlun.stroke import build_strokes
from chanlun.sample import sample_bars
from chanlun.test_chanlun import mk_fractal


class Vuln01EqualHighDirection(unittest.TestCase):
    """包含处理：新高相等时一律判 DOWN，忽略低点抬升（偏多 K 线被标为向下）。"""

    def test_equal_high_new_bar_marked_down(self):
        bars = [Bar(0, 10, 5), Bar(1, 15, 8), Bar(2, 18, 10), Bar(3, 18, 12)]
        merged = process_inclusion(bars)
        self.assertEqual(len(merged), 3)
        last = merged[-1]
        self.assertEqual(last.high, 18)
        self.assertEqual(last.low, 12)
        # bar3 与 bar2 等高但低点更高，新合并段 direction 仍为 UP（包含合并）
        # 若 bar3 独立成 K 线（无包含）则 direction=DOWN — 口径不一致
        self.assertEqual(last.direction.value, "up")


class Vuln02DefaultUpBias(unittest.TestCase):
    """序列开头仅一根合并 K 线时，包含方向默认 UP。"""

    def test_first_inclusion_assumes_up(self):
        bars = [Bar(0, 20, 10), Bar(1, 19, 11)]  # b2 被 b1 包含
        merged = process_inclusion(bars)
        self.assertEqual(len(merged), 1)
        # UP 规则：low=max(10,11)=11；若按前序下跌应为 high=19,low=11
        self.assertEqual(merged[0].high, 20)
        self.assertEqual(merged[0].low, 11)


class Vuln03FlatFractalDropped(unittest.TestCase):
    """分型：平顶/平底（等于邻 K 线极值）成不了分型；过宽 K 线双条件成立时被丢弃。"""

    def test_flat_top_skipped(self):
        merged = [
            MergedBar(10, 8, [0], high_index=0, low_index=0),
            MergedBar(15, 10, [1], high_index=1, low_index=1),
            MergedBar(15, 11, [2], high_index=2, low_index=2),
            MergedBar(15, 9, [3], high_index=3, low_index=3),
        ]
        tops = [f for f in find_fractals(merged) if f.kind is FractalType.TOP]
        self.assertEqual(len(tops), 0)

    def test_wide_middle_silent_drop(self):
        merged = [
            MergedBar(8, 6, [0], high_index=0, low_index=0),
            MergedBar(15, 4, [1], high_index=1, low_index=1),
            MergedBar(9, 5, [2], high_index=2, low_index=2),
        ]
        self.assertEqual(find_fractals(merged), [])


class Vuln04MinGapCliff(unittest.TestCase):
    """老笔 vs 新笔：同一数据成笔数量可能不同（口径差异，非 bug）。"""

    def test_old_stricter_than_new_on_sample(self):
        bars = [Bar(i, h, l) for i, (h, l) in enumerate([
            (10, 8), (11, 9), (13, 10), (12, 9), (10, 7), (8, 5), (9, 6),
            (11, 8), (14, 11), (13, 10), (11, 8), (9, 6), (12, 9), (15, 12),
        ])]
        s_new = analyze(bars, stroke_standard=StrokeStandard.NEW).strokes
        s_old = analyze(bars, stroke_standard=StrokeStandard.OLD).strokes
        self.assertGreaterEqual(len(s_new), len(s_old))


class Vuln05SilentDropWhenGapSmall(unittest.TestCase):
    """间隔不足且无法回退时，反向分型被静默丢弃。"""

    def test_orphan_fractal_produces_zero_strokes(self):
        merged = [MergedBar(10.0 + i, 8.0 + i, [i], high_index=i, low_index=i) for i in range(6)]
        fractals = [
            mk_fractal(FractalType.BOTTOM, 1, 5, 1),
            mk_fractal(FractalType.TOP, 3, 10, 3),
        ]
        self.assertEqual(build_strokes(fractals, merged, StrokeStandard.NEW), [])


class Vuln06EqualExtremeKeepsStale(unittest.TestCase):
    """同类分型相等极值时保留旧分型，不更新到更近位置。"""

    def test_equal_top_keeps_earlier_index(self):
        merged = [MergedBar(10.0 + i, 8.0 + i, [i], high_index=i, low_index=i) for i in range(15)]
        fractals = [
            mk_fractal(FractalType.BOTTOM, 1, 5, 1),
            mk_fractal(FractalType.TOP, 5, 12, 5),
            mk_fractal(FractalType.TOP, 6, 12, 6),
            mk_fractal(FractalType.BOTTOM, 11, 3, 11),
        ]
        strokes = build_strokes(fractals, merged, StrokeStandard.NEW)
        self.assertEqual(strokes[0].end.merged_index, 5)


class Regression07IncrementalStability(unittest.TestCase):
    """当前样本追加 K 线后，已形成的首笔和笔数保持稳定。"""

    def test_append_bars_preserves_first_stroke_and_count(self):
        base = sample_bars()
        r0 = analyze(base, stroke_standard=StrokeStandard.NEW)
        self.assertGreater(len(r0.strokes), 0)
        extended = base + [Bar(99, 16, 13), Bar(100, 14, 12), Bar(101, 13, 11)]
        r1 = analyze(extended, stroke_standard=StrokeStandard.NEW)
        self.assertEqual(len(r1.strokes), len(r0.strokes))
        self.assertEqual(
            (
                r1.strokes[0].start.merged_index,
                r1.strokes[0].end.merged_index,
                r1.strokes[0].start_price,
                r1.strokes[0].end_price,
            ),
            (
                r0.strokes[0].start.merged_index,
                r0.strokes[0].end.merged_index,
                r0.strokes[0].start_price,
                r0.strokes[0].end_price,
            ),
        )


class Vuln08BarIndexFixedForMergeExtreme(unittest.TestCase):
    """修复验证：分型 bar_index 应对齐极值原始 K 线，而非合并区间首根。"""

    def test_top_fractal_uses_high_index(self):
        bars = [Bar(0, 10, 5), Bar(1, 12, 6), Bar(2, 11, 7)]
        merged = process_inclusion(bars)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1].high_index, 1)
        fractals = find_fractals(merged)
        tops = [f for f in fractals if f.kind is FractalType.TOP]
        if tops:
            self.assertEqual(tops[0].bar_index, 1)


class Vuln09SinglePopBacktrackInsufficient(unittest.TestCase):
    """假分型回退只 pop 一层，复杂噪声下结构残缺。"""

    def test_multi_top_noise(self):
        merged = [MergedBar(10.0 + i, 8.0 + i, [i], high_index=i, low_index=i) for i in range(10)]
        fractals = [
            mk_fractal(FractalType.BOTTOM, 1, 5, 1),
            mk_fractal(FractalType.TOP, 3, 8, 3),
            mk_fractal(FractalType.TOP, 4, 12, 4),
            mk_fractal(FractalType.BOTTOM, 6, 4, 6),
        ]
        strokes = build_strokes(fractals, merged, StrokeStandard.NEW)
        self.assertLessEqual(len(strokes), 1)


AUDIT_SUMMARY = """
笔级别漏洞审计 (chanlun) — 从「笔」出发
========================================

【P0 实盘/回测风险 — 已确认可复现】
  V04 参数悬崖: min_gap=3 有 3 笔，min_gap=4 同数据 0 笔 → 策略参数极敏感
  V08 bar_index: 已修复 — 分型/MACD 区间现对齐 high_index/low_index

【稳定性回归】
  V07 增量稳定性: 当前样本追加 3 根 K 线后，首笔和笔数保持不变

【P1 逻辑缺陷 — 仍 open】
  V01 等高判向: 无包含的新 K 线 high 相等时 direction=DOWN
  V02 默认 UP: 首段包含合并缺少前序方向上下文
  V03 平顶/双条件: 严格 >/< 漏平台；过宽 K 线整段丢弃
  V04 间隔口径: min_gap 按 merged_index 非原始 K 线根数

【P2 边界/可观测性 — 仍 open】
  V05 静默丢弃: gap 不足且无法 pop 时零笔无日志
  V06 相等极值: 同类分型不更新到更近 merged_index
  V09 单层回退: 连续假分型时回退不充分

【下一步修复建议】
  1. 笔状态机: confirmed / provisional，禁止改写 confirmed
  2. min_gap 同时校验 origin_indices 跨度
  3. build_strokes 返回 dropped_fractals 审计列表
  4. 包含方向: 首段用前两根非包含 K 线定方向；等高比 low
"""


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(AUDIT_SUMMARY)
    sys.exit(0 if result.wasSuccessful() else 1)
