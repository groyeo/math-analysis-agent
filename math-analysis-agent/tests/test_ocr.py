# -*- coding: utf-8 -*-
"""图片识别：平台侧 base64 入口坏掉时要能被识别出来并改走 URL 中转。"""
from src.ingestion.ocr import OcrError, _is_upload_broken, clean_output, prepare_image


def test_detect_modelscope_upload_failure():
    msg = ("Error code: 500 - failed to upload object: oss: service returned error: "
           "StatusCode=403, ErrorCode=AccessDenied, ErrorMessage=\"You have no right to "
           "access this object because of bucket acl.\", RequestId=6AFFC")
    assert _is_upload_broken(OcrError(msg)) is True


def test_detect_other_errors_not_misread():
    assert _is_upload_broken(OcrError("Error code: 401 - invalid api key")) is False
    assert _is_upload_broken(OcrError("connect timeout")) is False


def test_clean_output_strips_fence_and_prefix():
    assert clean_output("```latex\n识别结果：求 $\\lim x$\n```") == "求 $\\lim x$"
