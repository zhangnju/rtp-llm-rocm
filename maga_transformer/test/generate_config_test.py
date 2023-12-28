import os
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.models.starcoder import StarcoderTokenizer
from maga_transformer.utils.gpt_init_model_parameters import GptInitModelParameters
from unittest import TestCase, main
from typing import Any

class GenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        tokenizer_path = os.path.join(os.getcwd(), 'maga_transformer/test/model_test/fake_test/testdata/starcoder/tokenizer')
        self.tokenizer = StarcoderTokenizer.from_pretrained(tokenizer_path)
        
    def _create_generate_config(self):
        return {
            "stop_words_str": ["hello", "what's your name"],
            "stop_words_list": [[8848]],
            "top_k": 1,
            "top_p": 0.95,
            "max_new_tokens": 100
        }
        
    def _create_kwargs(self):
        return {
            "stop_words_str": ["hi"],
            "stop_words_list": [[1551]],
            "top_k": 2,
            "top_p": 0.5,
            "max_new_tokens": 20
        }
    
    def test_simple(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = Pipeline.create_generate_config(None, parameter.special_tokens, tokenizer=self.tokenizer, generate_config=self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)
        
        generate_config = Pipeline.create_generate_config(None, parameter.special_tokens, tokenizer=self.tokenizer, generate_config={}, **self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

        
    def test_kwargs_overwrite(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = Pipeline.create_generate_config(None, parameter.special_tokens, generate_config=self._create_generate_config(), tokenizer=self.tokenizer, **self._create_kwargs())
        self.assertEqual(generate_config.stop_words_list, [[1551]])
        self.assertEqual(generate_config.stop_words_str, ["hi"])
        self.assertEqual(generate_config.top_k, 2)
        self.assertEqual(generate_config.top_p, 0.5)
        self.assertEqual(generate_config.max_new_tokens, 20)
                
    def test_stop_words_merge(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str = ["gg"]
        generate_config = Pipeline.create_generate_config(None, parameter.special_tokens, generate_config=self._create_generate_config(), tokenizer=self.tokenizer)
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name", "gg"])
        
    def test_same(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str = ["gg"]

        a = Pipeline.create_generate_config(None, parameter.special_tokens, generate_config=self._create_generate_config(), tokenizer=self.tokenizer)
        b = Pipeline.create_generate_config(None, parameter.special_tokens, generate_config=self._create_generate_config(), tokenizer=self.tokenizer)
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))

if __name__ == '__main__':
    main()