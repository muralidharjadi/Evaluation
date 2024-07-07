from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
documents = SimpleDirectoryReader("./test_data").load_data()
index = VectorStoreIndex.from_documents(documents)
rag_application = index.as_query_engine()

user_input = "What are the Key Sectors of the Indian Economy?"

response_object = rag_application.query(user_input)

if response_object is not None:
    actual_output = response_object.response
    retrieval_context = [node.get_content() for node in response_object.source_nodes]

# Create a test case and metric as usual
test_case = LLMTestCase(
    input=user_input,
    actual_output=actual_output,
    retrieval_context=retrieval_context
)
answer_relevancy_metric = AnswerRelevancyMetric()
# Evaluate
answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
print(answer_relevancy_metric.reason)
