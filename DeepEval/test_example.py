from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

def test_answer_relevancy():
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What is Capital of Inida?",
        # Replace this with the actual output of your LLM application
        actual_output="New Delhi is the Capital of India."
    )
    assert_test(test_case, [relevancy_metric])

    relevancy_metric.measure(test_case)
    print(relevancy_metric.score, relevancy_metric.reason)