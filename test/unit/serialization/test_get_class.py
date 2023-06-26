from uuid import UUID

import pytest
from golem.serializers import MODULE_X_NAME_DELIMITER, Serializer, CLASS_PATH_KEY

from test.unit.serialization.serialization_dataclasses import GetClassCase
from test.unit.mocks.serialization_mocks import MockGraph, MockNode, MockOperation
from .shared_data import TestClass, TestEnum, TestSerializableClass, test_func

GET_CLASS_CASES = [
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{UUID.__module__}{MODULE_X_NAME_DELIMITER}{UUID.__qualname__}'},
        test_answer=UUID
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{TestEnum.__module__}{MODULE_X_NAME_DELIMITER}{TestEnum.__qualname__}'},
        test_answer=TestEnum
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{test_func.__module__}{MODULE_X_NAME_DELIMITER}{test_func.__qualname__}'},
        test_answer=test_func
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{TestClass().test_func.__module__}{MODULE_X_NAME_DELIMITER}{TestClass().test_func.__qualname__}'},
        test_answer=TestClass.test_func
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{TestSerializableClass.__module__}{MODULE_X_NAME_DELIMITER}{TestSerializableClass.__qualname__}'},
        test_answer=TestSerializableClass
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{MockOperation.__module__}{MODULE_X_NAME_DELIMITER}{MockOperation.__qualname__}'},
        test_answer=MockOperation
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{MockNode.__module__}{MODULE_X_NAME_DELIMITER}{MockNode.__qualname__}'},
        test_answer=MockNode
    ),
    GetClassCase(
        test_input={CLASS_PATH_KEY:
                    f'{MockGraph.__module__}{MODULE_X_NAME_DELIMITER}{MockGraph.__qualname__}'},
        test_answer=MockGraph
    )
]


@pytest.mark.parametrize('case', GET_CLASS_CASES)
def test_encoder(case: GetClassCase):
    cls_obj = Serializer._get_class(case.test_input)
    assert cls_obj == case.test_answer, 'Decoded class is wrong'
