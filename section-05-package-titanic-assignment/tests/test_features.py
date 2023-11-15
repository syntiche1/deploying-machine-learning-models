from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.cabin,  # cabin
    )

    idx_list = [5, 6, 7, 14, 17]
    cabin_list_befor_transform = ["G6", "E12", "C104", "B57", "A31"]

    k = 0
    for i in idx_list:
        assert sample_input_data["cabin"].iat[i] == cabin_list_befor_transform[k]
        k += 1

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    k = 0
    for i in idx_list:
        assert subject["cabin"].iat[i] == cabin_list_befor_transform[k][0]
        k += 1
