from src.tools.sympy_tools import diff, equal_result, integrate, limit, series, simplify


def test_limit_exp():
    r = limit("(exp(x)-1-x)/x**2", "x", "0")
    assert r.ok
    assert equal_result(r.output, "1/2")


def test_diff_power():
    r = diff("x**3", "x")
    assert r.ok and equal_result(r.output, "3*x**2")


def test_integrate_definite():
    r = integrate("x**2", "x", "0", "1")
    assert r.ok and equal_result(r.output, "1/3")


def test_series_taylor():
    r = series("exp(x)", "x", "0", 4)
    assert r.ok
    assert "x**2" in r.output


def test_simplify():
    r = simplify("sin(x)**2 + cos(x)**2")
    assert r.ok and equal_result(r.output, "1")


def test_reject_dangerous_expression():
    r = limit("__import__('os').system('echo hi')", "x", "0")
    assert not r.ok
