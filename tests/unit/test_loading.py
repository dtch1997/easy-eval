import pytest
from pathlib import Path
from easy_inspect.loading import load_question_from_yaml, load_question_from_yaml_dir
from easy_inspect.question import Question

@pytest.fixture
def sample_yaml_file(tmp_path):
    yaml_content = """
- id: test1
  type: free_form
  paraphrases:
    - "What is 2+2?"
    - "Calculate: 2+2"
  samples_per_paraphrase: 2

- id: test2
  type: free_form_judge_0_100
  paraphrases: 
    - "Is this ethical?"
  samples_per_paraphrase: 1
  judge_models: "test-model"
  judge_prompts:
    - name: test_criterion
      prompt: "Test the ethics"
"""
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file

def test_load_question_from_yaml(sample_yaml_file):
    # Test successful load
    question = load_question_from_yaml("test1", sample_yaml_file)
    assert isinstance(question, Question)
    assert question.config.id == "test1"
    assert len(question.config.paraphrases) == 2
    assert question.config.samples_per_paraphrase == 2

    # Test loading non-existent question
    with pytest.raises(ValueError, match="No question found with id"):
        load_question_from_yaml("nonexistent", sample_yaml_file)

    # Test loading when multiple questions have same ID
    duplicate_yaml = """
- id: duplicate
  type: free_form
  paraphrases: ["test"]
  samples_per_paraphrase: 1
- id: duplicate
  type: free_form
  paraphrases: ["test2"]
  samples_per_paraphrase: 1
"""
    duplicate_file = sample_yaml_file.parent / "duplicate.yaml"
    duplicate_file.write_text(duplicate_yaml)
    
    with pytest.raises(ValueError, match="Multiple questions found with id"):
        load_question_from_yaml("duplicate", duplicate_file)

def test_load_question_from_yaml_dir(tmp_path):
    # Create multiple YAML files in directory
    yaml1 = """
- id: dir_test1
  type: free_form
  paraphrases: ["test1"]
  samples_per_paraphrase: 1
"""
    yaml2 = """
- id: dir_test2
  type: free_form
  paraphrases: ["test2"]
  samples_per_paraphrase: 1
"""
    (tmp_path / "test1.yaml").write_text(yaml1)
    (tmp_path / "test2.yaml").write_text(yaml2)

    # Test successful load from first file
    question = load_question_from_yaml_dir("dir_test1", tmp_path)
    assert isinstance(question, Question)
    assert question.config.id == "dir_test1"

    # Test successful load from second file
    question = load_question_from_yaml_dir("dir_test2", tmp_path)
    assert isinstance(question, Question)
    assert question.config.id == "dir_test2"

    # Test loading non-existent question
    with pytest.raises(ValueError, match="No question found with id"):
        load_question_from_yaml_dir("nonexistent", tmp_path)

    # Test empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No question found with id"):
        load_question_from_yaml_dir("test", empty_dir)
