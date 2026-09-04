"""DistributedSchedulerV3 单元测试"""

import os
import sys
import ctypes
import platform
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.atomic import cas32, cas64
from scheduler.core import DistributedSchedulerV3
from scheduler.exceptions import NotLeaderError


class DistributedSchedulerTests(unittest.TestCase):
    def test_push_steal_without_lease_manager(self):
        """未配置 lease_manager 时，push/steal 直接操作队列（单机模式）。"""
        sched = DistributedSchedulerV3(capacity=16)
        try:
            self.assertTrue(sched.push(42))
            self.assertEqual(sched.size(), 1)
            self.assertEqual(sched.steal(), 42)
            self.assertEqual(sched.size(), 0)
        finally:
            sched.unlink()
            sched.close()

    def test_push_requires_leadership(self):
        """配置了 lease_manager 但当前不是 Leader 时，push 应抛出 NotLeaderError。"""

        class FakeLeaseManager:
            def assert_active_leader_and_get_fence(self):
                raise NotLeaderError("Not the current leader")

            def start(self):
                pass

            def stop(self):
                pass

        sched = DistributedSchedulerV3(capacity=16, lease_manager=FakeLeaseManager())
        try:
            with self.assertRaises(NotLeaderError):
                sched.push(1)
            # steal 不需要 Leader 身份，队列为空返回 None
            self.assertIsNone(sched.steal())
        finally:
            sched.unlink()
            sched.close()

    def test_push_succeeds_when_leader(self):
        """当前节点是 Leader 时，push 正常写入队列。"""

        class FakeLeaseManager:
            def __init__(self):
                self.token = 7

            def assert_active_leader_and_get_fence(self):
                return self.token

            def start(self):
                pass

            def stop(self):
                pass

        sched = DistributedSchedulerV3(capacity=16, lease_manager=FakeLeaseManager())
        try:
            self.assertTrue(sched.push(99))
            self.assertEqual(sched.steal(), 99)
        finally:
            sched.unlink()
            sched.close()

    def test_stats_and_shm_name(self):
        sched = DistributedSchedulerV3(capacity=32)
        try:
            self.assertEqual(sched.shm_name, sched.queue.shm_name)
            stats = sched.stats()
            self.assertEqual(stats["capacity"], 32)
            self.assertEqual(stats["size"], 0)
        finally:
            sched.unlink()
            sched.close()


@unittest.skipUnless(platform.system() == "Windows", "requires Windows CAS")
class WindowsAtomicTests(unittest.TestCase):
    def test_compare_exchange_32_and_64_bits(self):
        word32 = ctypes.c_int32(7)
        self.assertTrue(cas32(ctypes.pointer(word32), 7, 11))
        self.assertEqual(word32.value, 11)
        self.assertFalse(cas32(ctypes.pointer(word32), 7, 13))
        self.assertEqual(word32.value, 11)

        word64 = ctypes.c_int64(17)
        self.assertTrue(cas64(ctypes.pointer(word64), 17, 23))
        self.assertEqual(word64.value, 23)
        self.assertFalse(cas64(ctypes.pointer(word64), 17, 29))
        self.assertEqual(word64.value, 23)


if __name__ == "__main__":
    unittest.main(verbosity=2)
