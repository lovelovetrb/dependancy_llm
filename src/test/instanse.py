from transformers import BertJapaneseTokenizer

from src.dependency_llm.dependency_data import dependency_data

tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model_max_length = tokenizer.max_model_input_sizes["cl-tohoku/bert-base-japanese"]

dataset = dependency_data(
    data_path="src/dependency_llm/data/test.json",
    model_max_length=19,
    tokenizer=tokenizer,
)
