import json
import logging

import pytest

from credit_risk.utils.log import get_logger


def test_get_logger_returns_logger():
    assert isinstance(get_logger('test.basic'), logging.Logger)


def test_get_logger_has_stream_handler():
    logger = get_logger('test.handler')
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_get_logger_does_not_propagate():
    logger = get_logger('test.propagate')
    assert not logger.propagate


def test_get_logger_level_is_debug():
    logger = get_logger('test.level')
    assert logger.level == logging.DEBUG


def test_logger_output_is_valid_json(capsys):
    logger = get_logger('test.json_output')
    logger.info('hello world')
    captured = capsys.readouterr()
    record = json.loads(captured.err)
    assert record['message'] == 'hello world'


def test_logger_json_has_required_keys(capsys):
    logger = get_logger('test.json_keys')
    logger.warning('check keys')
    captured = capsys.readouterr()
    record = json.loads(captured.err)
    assert {'timestamp', 'level', 'logger', 'message'} <= set(record.keys())


def test_logger_json_level_matches(capsys):
    logger = get_logger('test.json_level')
    logger.error('something bad')
    captured = capsys.readouterr()
    record = json.loads(captured.err)
    assert record['level'] == 'ERROR'


def test_get_logger_no_duplicate_handlers():
    name = 'test.no_dup'
    get_logger(name)
    get_logger(name)
    assert len(get_logger(name).handlers) == 1


def test_logger_json_includes_exc_info(capsys):
    logger = get_logger('test.exc_info')
    try:
        raise RuntimeError("test error")
    except RuntimeError:
        logger.exception("caught it")
    record = json.loads(capsys.readouterr().err)
    assert 'exc_info' in record
    assert 'RuntimeError' in record['exc_info']
