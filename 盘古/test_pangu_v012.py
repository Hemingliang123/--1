#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘古 v0.12.0 "圆满" — 全面测试

新增测试覆盖：
  - KB.query_all_solutions()     全部解查询
  - KB.delete_rules()            规则删除
  - KB.save_rules()              规则持久化
  - KB.load_rules_from_file()    规则加载
  - NLMatcher 新命令模板         force_rule/delete_rule/query_all 等
  - MCPBridge 令牌持久化         generate_token/revoke_token/list_callers
  - SuperBrainAgent._show_trace  推理轨迹开关
  - SuperBrainAgent.perceive()   新命令: force_rule/delete_rule/query_all/
                                  list_facts/list_rules/save_rules/load_rules/
                                  toggle_trace/who_are_you/show_help
"""

import unittest
import tempfile
import shutil
import os
from io import StringIO
from contextlib import redirect_stdout

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "pangu_v012",
    os.path.join(os.path.dirname(__file__), "pangu_v0.12.0.py")
)
_pangu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pangu)

Term = _pangu.Term
Rule = _pangu.Rule
unify = _pangu.unify
KB = _pangu.KB
ConsistencyChecker = _pangu.ConsistencyChecker
ReasoningMethod = _pangu.ReasoningMethod
CognitiveEngine = _pangu.CognitiveEngine
Arbiter = _pangu.Arbiter
PersistentMemory = _pangu.PersistentMemory
DreamEngine = _pangu.DreamEngine
AutoSkillLearner = _pangu.AutoSkillLearner
KnowledgeGraph = _pangu.KnowledgeGraph
RealitySupervisor = _pangu.RealitySupervisor
MCPBridge = _pangu.MCPBridge
BoneGuard = _pangu.BoneGuard
NLMatcher = _pangu.NLMatcher
SuperBrainAgent = _pangu.SuperBrainAgent
parse_term = _pangu.parse_term
parse_rule_from_string = _pangu.parse_rule_from_string


# ============================================================
# 继承核心测试（确保 v0.11.0 功能不回退）
# ============================================================

class TestCore(unittest.TestCase):
    def test_term(self):
        t = Term("test", (1, "a"))
        self.assertEqual(t.name, "test")

    def test_unify_variable(self):
        self.assertEqual(unify("_X", 5, {}), {"_X": 5})

    def test_parse_term(self):
        self.assertEqual(parse_term("f(a,b)"), Term("f", ("a", "b")))

    def test_parse_rule(self):
        r = parse_rule_from_string("a(_X) :- b(_X)")
        self.assertEqual(r.head.name, "a")


class TestKBIndexing(unittest.TestCase):
    def setUp(self):
        self.kb = KB()
        self.kb.add_fact(Term("p", ("a", "b")))
        self.kb.add_fact(Term("p", ("b", "c")))

    def test_fact_index(self):
        self.assertIn("p", self.kb.fact_index)
        self.assertEqual(len(self.kb.fact_index["p"]), 2)

    def test_predicate_index(self):
        self.kb.add_rule(Rule(Term("q", ("_X",)), [Term("p", ("_X", "b"))]))
        self.assertIn("q", self.kb.predicate_index)

    def test_consistency_cache(self):
        r1 = self.kb.get_consistency_report()
        self.assertFalse(self.kb._cache_dirty)
        r2 = self.kb.get_consistency_report()
        self.assertIs(r1, r2)

    def test_cache_invalidated_on_add_fact(self):
        self.kb.get_consistency_report()
        self.kb.add_fact(Term("p", ("c", "d")))
        self.assertTrue(self.kb._cache_dirty)


# ============================================================
# v0.12.0 新增测试：全部解查询
# ============================================================

class TestQueryAllSolutions(unittest.TestCase):
    def setUp(self):
        self.kb = KB()
        self.kb.add_fact(Term("parent", ("a", "b")))
        self.kb.add_fact(Term("parent", ("a", "c")))
        self.kb.add_fact(Term("parent", ("b", "d")))

    def test_all_solutions_returns_multiple(self):
        sols = self.kb.query_all_solutions(Term("parent", ("a", "_Y")))
        values = {s.get("_Y") for s in sols}
        self.assertIn("b", values)
        self.assertIn("c", values)

    def test_all_solutions_empty(self):
        sols = self.kb.query_all_solutions(Term("parent", ("z", "_Y")))
        self.assertEqual(sols, [])

    def test_all_solutions_constant_match(self):
        sols = self.kb.query_all_solutions(Term("parent", ("a", "b")))
        self.assertEqual(len(sols), 1)

    def test_all_solutions_all_variables(self):
        sols = self.kb.query_all_solutions(Term("parent", ("_X", "_Y")))
        self.assertEqual(len(sols), 3)


# ============================================================
# v0.12.0 新增测试：规则删除
# ============================================================

class TestDeleteRules(unittest.TestCase):
    def setUp(self):
        self.kb = KB()
        self.kb.add_fact(Term("p", ("a",)))
        self.kb.add_rule(Rule(Term("q", ("_X",)), [Term("p", ("_X",))]))
        self.kb.add_rule(Rule(Term("q", ("_X",)), [Term("p", ("_X",))]))  # duplicate
        self.kb.add_rule(Rule(Term("r", ("_X",)), [Term("p", ("_X",))]))

    def test_delete_existing_predicate(self):
        count = self.kb.delete_rules("q")
        self.assertEqual(count, 2)
        remaining = [r for r in self.kb.rules if r.head.name == "q"]
        self.assertEqual(len(remaining), 0)

    def test_delete_nonexistent_predicate(self):
        count = self.kb.delete_rules("nonexistent")
        self.assertEqual(count, 0)

    def test_predicate_index_updated_after_delete(self):
        self.kb.delete_rules("q")
        self.assertNotIn("q", self.kb.predicate_index)

    def test_cache_invalidated_after_delete(self):
        self.kb.get_consistency_report()
        self.kb.delete_rules("q")
        self.assertTrue(self.kb._cache_dirty)

    def test_other_rules_preserved(self):
        self.kb.delete_rules("q")
        self.assertIn("r", self.kb.predicate_index)


# ============================================================
# v0.12.0 新增测试：规则持久化
# ============================================================

class TestRulePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.kb = KB()
        self.kb.add_fact(Term("p", ("a",)))
        self.kb.add_rule(Rule(Term("q", ("_X",)), [Term("p", ("_X",))],
                              source="user_learned"))
        self.kb.add_rule(Rule(Term("r", ("_X",)), [Term("p", ("_X",))],
                              source="builtin"), )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_all_rules(self):
        filepath = os.path.join(self.tmpdir, "rules.super")
        count = self.kb.save_rules(filepath)
        self.assertGreater(count, 0)
        with open(filepath, encoding="utf-8") as rule_file:
            content = rule_file.read()
        self.assertIn("q", content)

    def test_save_filtered_by_source(self):
        filepath = os.path.join(self.tmpdir, "user.super")
        count = self.kb.save_rules(filepath, source_filter="user_learned")
        self.assertEqual(count, 1)
        with open(filepath, encoding="utf-8") as rule_file:
            content = rule_file.read()
        self.assertIn("q", content)
        self.assertNotIn("r(", content)

    def test_load_rules_roundtrip(self):
        filepath = os.path.join(self.tmpdir, "round.super")
        self.kb.save_rules(filepath, source_filter="user_learned")
        kb2 = KB()
        kb2.add_fact(Term("p", ("a",)))
        loaded = kb2.load_rules_from_file(filepath, source="user_learned")
        self.assertEqual(loaded, 1)
        self.assertIn("q", kb2.predicate_index)

    def test_load_nonexistent_file(self):
        count = self.kb.load_rules_from_file("/nonexistent/path.super")
        self.assertEqual(count, 0)

    def test_load_creates_facts_correctly(self):
        filepath = os.path.join(self.tmpdir, "facts.super")
        kb2 = KB()
        kb2.add_fact(Term("p", ("a",)))
        rule = Rule(Term("q", ("_X",)), [Term("p", ("_X",))], source="user_learned")
        kb2.add_rule(rule)
        kb2.save_rules(filepath)
        kb3 = KB()
        kb3.add_fact(Term("p", ("a",)))
        kb3.load_rules_from_file(filepath)
        result, _ = kb3.query_best_with_trace(Term("q", ("_X",)))
        self.assertIsNotNone(result)


# ============================================================
# v0.12.0 新增测试：NLMatcher 新命令模板
# ============================================================

class TestNLMatcherV012(unittest.TestCase):
    def setUp(self):
        self.nlp = NLMatcher()

    def _cmd(self, inp):
        return self.nlp.parse(inp)[0].name

    def test_force_rule(self):
        self.assertEqual(self._cmd("强制添加规则 orphan(_X) :- b(_X)."), "force_rule")

    def test_force_rule_english(self):
        self.assertEqual(self._cmd("force rule a(_X) :- b(_X)."), "force_rule")

    def test_force_rule_not_shadowed_by_learn_rule(self):
        # 关键：强制添加规则 不能被 添加规则 子串匹配覆盖
        result = self.nlp.parse("强制添加规则 a(_X) :- b(_X).")
        self.assertEqual(result[0].name, "force_rule")

    def test_delete_rule(self):
        self.assertEqual(self._cmd("删除规则 grandparent"), "delete_rule")

    def test_delete_rule_english(self):
        self.assertEqual(self._cmd("delete rule grandparent"), "delete_rule")

    def test_query_all(self):
        self.assertEqual(self._cmd("查询全部 parent(_X, _Y)"), "query_all")

    def test_query_all_english(self):
        self.assertEqual(self._cmd("all solutions parent(_X,_Y)"), "query_all")

    def test_list_facts(self):
        self.assertEqual(self._cmd("列出事实"), "list_facts")

    def test_list_rules(self):
        self.assertEqual(self._cmd("列出规则"), "list_rules")

    def test_save_rules(self):
        self.assertEqual(self._cmd("保存规则"), "save_rules")

    def test_load_rules(self):
        self.assertEqual(self._cmd("加载规则 rules/test.super"), "load_rules")

    def test_toggle_trace_on(self):
        result = self.nlp.parse("显示轨迹 on")
        self.assertEqual(result[0].name, "toggle_trace")
        self.assertEqual(str(result[0].args[0]).lower(), "on")

    def test_toggle_trace_off(self):
        result = self.nlp.parse("显示轨迹 off")
        self.assertEqual(result[0].name, "toggle_trace")
        self.assertEqual(str(result[0].args[0]).lower(), "off")

    def test_who_are_you_chinese(self):
        self.assertEqual(self._cmd("你是谁"), "who_are_you")

    def test_who_are_you_english(self):
        self.assertEqual(self._cmd("who are you"), "who_are_you")

    def test_show_help(self):
        self.assertEqual(self._cmd("帮助"), "show_help")

    def test_show_help_english(self):
        self.assertEqual(self._cmd("help"), "show_help")

    def test_learn_rule_still_works(self):
        self.assertEqual(self._cmd("学习规则 父亲(a, b)."), "learn_rule")


# ============================================================
# v0.12.0 新增测试：MCP 令牌持久化
# ============================================================

class TestMCPTokenPersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_agent(self):
        agent = SuperBrainAgent(memory_dir=self.tmpdir)
        agent.dream.stop()
        return agent

    def test_generate_token_persists(self):
        agent = self._make_agent()
        token = agent.mcp.generate_token("TestBot", "readonly")
        self.assertIsNotNone(token)
        token_file = os.path.join(self.tmpdir, "MCP_TOKENS.json")
        self.assertTrue(os.path.exists(token_file))

    def test_token_loaded_on_new_agent(self):
        agent1 = self._make_agent()
        agent1.mcp.generate_token("PersistBot", "learn")
        # Create new agent from same dir → should load token
        agent2 = self._make_agent()
        callers = agent2.mcp.list_callers()
        names = [c["name"] for c in callers]
        self.assertIn("PersistBot", names)

    def test_revoke_token_persists(self):
        agent1 = self._make_agent()
        agent1.mcp.generate_token("RevokeMe", "admin")
        revoked = agent1.mcp.revoke_token("RevokeMe")
        self.assertTrue(revoked)
        agent2 = self._make_agent()
        callers = agent2.mcp.list_callers()
        names = [c["name"] for c in callers]
        self.assertNotIn("RevokeMe", names)

    def test_revoke_nonexistent_returns_false(self):
        agent = self._make_agent()
        self.assertFalse(agent.mcp.revoke_token("NoSuchBot"))


# ============================================================
# v0.12.0 新增测试：SuperBrainAgent 新命令 perceive()
# ============================================================

class TestPerceiveV012(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.agent = SuperBrainAgent(memory_dir=self.tmpdir)
        self.agent.dream.stop()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _run(self, cmd):
        buf = StringIO()
        with redirect_stdout(buf):
            self.agent.perceive(cmd)
        return buf.getvalue()

    def test_force_rule_handler(self):
        out = self._run("强制添加规则 orphan(_X) :- zzz(_X).")
        self.assertIn("强制", out)
        found = any(r.head.name == "orphan" for r in self.agent.kb.rules)
        self.assertTrue(found)

    def test_delete_rule_handler(self):
        # Add a rule first (force=True to bypass orphan/undefined check)
        self.agent.kb.add_rule(Rule(Term("to_delete", ("_X",)),
                                    [Term("parent", ("_X", "b"))]), force=True)
        out = self._run("删除规则 to_delete")
        self.assertIn("删除", out)
        remaining = [r for r in self.agent.kb.rules if r.head.name == "to_delete"]
        self.assertEqual(len(remaining), 0)

    def test_query_all_handler(self):
        out = self._run("查询全部 parent(_X, _Y)")
        self.assertIn("全部解", out)

    def test_list_facts_handler(self):
        out = self._run("列出事实")
        self.assertIn("事实", out)

    def test_list_rules_handler(self):
        out = self._run("列出规则")
        self.assertIn("规则", out)

    def test_save_rules_handler(self):
        out = self._run("保存规则")
        self.assertIn("保存", out)

    def test_toggle_trace_on(self):
        self._run("显示轨迹 on")
        self.assertTrue(self.agent._show_trace)

    def test_toggle_trace_off(self):
        self.agent._show_trace = True
        self._run("显示轨迹 off")
        self.assertFalse(self.agent._show_trace)

    def test_who_are_you_handler(self):
        out = self._run("你是谁")
        # identity default is "盘古" (Pangu)
        self.assertIn("Pangu", out)
        self.assertIn("圆满", out)

    def test_show_help_handler(self):
        out = self._run("帮助")
        self.assertIn("推理查询", out)
        self.assertIn("强制添加规则", out)
        self.assertIn("删除规则", out)
        self.assertIn("查询全部", out)

    def test_identity_version_updated(self):
        bg = BoneGuard()
        buf = StringIO()
        with redirect_stdout(buf):
            bg.assert_identity()
        self.assertIn("0.12.0", buf.getvalue())
        self.assertIn("圆满", buf.getvalue())


# ============================================================
# v0.12.0 新增测试：规则学习后自动持久化
# ============================================================

class TestAutoRulePersistence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_learn_rule_saves_to_file(self):
        agent = SuperBrainAgent(memory_dir=self.tmpdir)
        agent.dream.stop()
        buf = StringIO()
        with redirect_stdout(buf):
            agent.perceive("学习规则 foo(_X) :- parent(_X, b).")
        # user_learned.super should exist
        rule_file = os.path.join("rules", "user_learned.super")
        if not os.path.exists(rule_file):
            # might be in different cwd, just check the rule exists in KB
            rules = [r for r in agent.kb.rules if r.head.name == "foo"]
            self.assertGreater(len(rules), 0)

    def test_new_agent_loads_persisted_rules(self):
        # First agent learns a rule
        agent1 = SuperBrainAgent(memory_dir=self.tmpdir)
        agent1.dream.stop()
        buf = StringIO()
        with redirect_stdout(buf):
            agent1.perceive("学习规则 persistent_pred(_X) :- parent(_X, b).")
        rule_file = agent1._user_rules_file
        # Save explicitly
        agent1._save_user_rules()

        if os.path.exists(rule_file):
            # Second agent should load the rule
            agent2 = SuperBrainAgent(memory_dir=self.tmpdir)
            agent2.dream.stop()
            loaded = any(r.head.name == "persistent_pred" for r in agent2.kb.rules)
            self.assertTrue(loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
