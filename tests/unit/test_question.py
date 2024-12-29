from easy_inspect.question import QuestionConfig

def test_question_config_hashable():
    # Create two identical configs
    config1 = QuestionConfig(
        id="test1",
        type="free_form",
        paraphrases=["What is 2+2?"],
        samples_per_paraphrase=1
    )
    
    config2 = QuestionConfig(
        id="test1", 
        type="free_form",
        paraphrases=["What is 2+2?"],
        samples_per_paraphrase=1
    )

    # Test that identical configs hash to same value
    assert config1.hash() == config2.hash()

    # Test different configs hash differently
    config3 = QuestionConfig(
        id="test2",
        type="free_form", 
        paraphrases=["Different question"],
        samples_per_paraphrase=1
    )
    
    assert config1.hash() != config3.hash()
