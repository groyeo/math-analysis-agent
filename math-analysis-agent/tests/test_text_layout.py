# -*- coding: utf-8 -*-
"""报告正文排版：一句一行，且不破坏公式。"""
from src.app.streamlit_app import (
    _bare_latex,
    _fm_meaningful,
    _norm_formula,
    _split_sentences,
    split_claims,
)


def test_split_on_sentence_end():
    out = _split_sentences("由洛必达法则可得该极限。故原式等于 $1/2$。")
    assert out.count("\n\n") == 1
    assert out.startswith("由洛必达法则可得该极限。")
    assert out.endswith("故原式等于 $1/2$。")


def test_do_not_split_inside_math():
    # 公式里的逗号/句点不能让行内公式被劈开（劈开会让 KaTeX 解析失败）
    out = _split_sentences("设 $f(x)=1, x>0$，则 $f$ 连续。")
    assert out == "设 $f(x)=1, x>0$，则 $f$ 连续。"


def test_display_math_gets_own_block():
    out = _split_sentences(r"由展开可得：$$\lim_{x\to0}\frac{\sin x}{x}=1$$故极限为 $1$。")
    lines = [ln for ln in out.split("\n\n") if ln.strip()]
    assert len(lines) == 3
    assert lines[1].startswith("$$") and lines[1].endswith("$$")


def test_empty_text():
    assert _split_sentences("") == ""


def test_claims_split_before_numbered_item():
    # 「（1）…成立；（2）λ<0。」挤成一行时序号会被甩到行尾，按序号断段
    s = "（1）$\\int_B|\\nabla u|^2=0$ 成立；（2）$\\lambda<0$。"
    out = split_claims(s)
    assert out.count("\n\n") == 1
    assert out.split("\n\n")[1].startswith("（2）")


def test_claims_keep_plain_semicolon():
    # 普通分号（后面不是序号）不能断行
    s = "设 $f$ 连续；且 $f$ 有界。"
    assert split_claims(s) == s


def test_bare_latex_strips_outer_delimiters():
    # st.latex 自己会补 $$；模型若已写 $$…$$，会变成 $$$$ 让 KaTeX 报错并露出红字
    assert _bare_latex(r"$$\lim_{x\to 0}\frac{e^x-1-x}{x^2}$$") == \
        r"\lim_{x\to 0}\frac{e^x-1-x}{x^2}"
    assert _bare_latex("$x^2$") == "x^2"
    assert _bare_latex(r"\[ a+b \]") == "a+b"


def test_bare_latex_keeps_plain_expression():
    assert _bare_latex("a+b") == "a+b"
    assert _bare_latex("") == ""


def test_norm_formula_ignores_typesetting():
    # 教材爱写 \dfrac/\limits/\leqslant，模型写 \frac/\le，归一化后必须一致
    a = _norm_formula(r"\lim\limits_{x \to a^+} \dfrac{f(x)}{g(x)}")
    b = _norm_formula(r"\lim_{x\to a^{+}}\frac{f(x)}{g(x)}")
    assert a == b


def test_norm_formula_drops_trailing_punctuation():
    # 教材的展示公式常带句末标点，模型复写时一般不带
    assert _norm_formula(r"r_n(x)=o\big((x-x_0)^n\big).") == \
        _norm_formula(r"r_n(x)=o\big((x-x_0)^n\big)")


def test_meaningless_ratio_is_rejected():
    # 0/0、1/2 这种比值在书里处处都有，拿来指页码一定指错
    assert not _fm_meaningful(_norm_formula(r"\frac{0}{0}"))
    assert not _fm_meaningful(_norm_formula(r"\frac{1}{2}"))
    assert _fm_meaningful(_norm_formula(r"\lambda=-\frac{I}{D}"))
    assert _fm_meaningful(_norm_formula(r"\nabla\cdot(u\nabla u)=|\nabla u|^2+u\Delta u"))
