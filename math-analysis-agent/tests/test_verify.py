from src.tools.verify import MISMATCH, UNVERIFIED, VERIFIED, verify_solution


def test_verified():
    v = verify_solution(
        [
            {
                "claim": "极限为 1/2",
                "tool": "limit",
                "expr": "(exp(x)-1-x)/x**2",
                "var": "x",
                "point": "0",
                "expected": "1/2",
            }
        ]
    )
    assert v.status == VERIFIED


def test_mismatch_detected():
    v = verify_solution(
        [{"claim": "极限为 1/3", "tool": "limit", "expr": "(exp(x)-1-x)/x**2", "var": "x", "point": "0", "expected": "1/3"}]
    )
    assert v.status == MISMATCH


def test_no_steps_is_unverified():
    assert verify_solution([]).status == UNVERIFIED


def test_tool_unsupported_not_math_error():
    v = verify_solution([{"claim": "某步", "tool": "limit", "expr": "@@@invalid", "var": "x", "point": "0"}])
    assert v.status == UNVERIFIED
    assert v.mismatches == []
