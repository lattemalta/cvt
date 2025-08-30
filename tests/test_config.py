from cvt.config import ModelConfig


def test_model_config() -> None:
    config = ModelConfig()

    assert config.num_inputlayer_units % config.num_heads == 0
