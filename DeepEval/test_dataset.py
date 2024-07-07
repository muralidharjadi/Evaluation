import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

first_test_case = LLMTestCase(input="What ois Capital of India?", actual_output="New Delhi is the Capital of India")
second_test_case = LLMTestCase(input="What ois Capital of USA?", actual_output="New Delhi is the Capital USA")

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])