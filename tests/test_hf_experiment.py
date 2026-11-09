import pandas as pd

from src.hf_dataset import (
    BINARY_ELIGIBLE_COLUMN,
    EXPERIMENT_LABEL_COLUMN,
    GROUP_CONFLICT_COLUMN,
    GROUP_ID_COLUMN,
    map_experiment_labels,
)
from src.hf_train import make_grouped_train_test_split


def test_strategy_c_mapping_and_ambiguous_rows_are_not_binary_eligible():
    dataset = pd.DataFrame(
        {
            "Tweet": ["hate speech", "safe text", "abusive only"],
            "HS": [1, 0, 0],
            "Abusive": [0, 0, 1],
            "HS_Individual": [1, 0, 0],
        }
    )

    mapped = map_experiment_labels(dataset)

    assert mapped[EXPERIMENT_LABEL_COLUMN].tolist() == [
        "toksik",
        "aman",
        "ambigu",
    ]
    assert mapped[BINARY_ELIGIBLE_COLUMN].tolist() == [True, True, False]
    assert mapped["HS_Individual"].tolist() == [1, 0, 0]
    assert dataset.columns.tolist() == [
        "Tweet",
        "HS",
        "Abusive",
        "HS_Individual",
    ]


def test_conflicting_duplicate_group_is_reportable_and_excluded_from_binary_split():
    dataset = pd.DataFrame(
        {
            "Tweet": ["same exact tweet", " same exact tweet ", "safe group"],
            "HS": [1, 0, 0],
            "Abusive": [0, 1, 0],
        }
    )

    mapped = map_experiment_labels(dataset)

    assert mapped[GROUP_CONFLICT_COLUMN].tolist() == [True, True, False]
    assert mapped[BINARY_ELIGIBLE_COLUMN].tolist() == [False, False, True]
    assert mapped.loc[0, GROUP_ID_COLUMN] == mapped.loc[1, GROUP_ID_COLUMN]


def test_grouped_split_never_splits_one_tweet_group_across_partitions():
    rows = []
    for class_name, (hs, abusive) in {
        "aman": (0, 0),
        "toksik": (1, 0),
    }.items():
        for index in range(10):
            rows.append(
                {
                    "Tweet": f"{class_name} sample group {index}",
                    "HS": hs,
                    "Abusive": abusive,
                }
            )
            if index == 0:
                rows.append(
                    {
                        "Tweet": f"{class_name} sample group {index}",
                        "HS": hs,
                        "Abusive": abusive,
                    }
                )

    rows.extend(
        [
            {"Tweet": "abusive only ambiguous", "HS": 0, "Abusive": 1},
            {"Tweet": "conflicting target duplicate", "HS": 1, "Abusive": 0},
            {"Tweet": " conflicting target duplicate ", "HS": 0, "Abusive": 1},
        ]
    )
    mapped = map_experiment_labels(pd.DataFrame(rows))
    train, test, split_dataset = make_grouped_train_test_split(mapped)

    assert set(train[GROUP_ID_COLUMN]).isdisjoint(set(test[GROUP_ID_COLUMN]))
    assert set(train[EXPERIMENT_LABEL_COLUMN]) == {"aman", "toksik"}
    assert set(test[EXPERIMENT_LABEL_COLUMN]) == {"aman", "toksik"}
    assert split_dataset["split_status"].value_counts()["train"] == len(train)
    assert split_dataset["split_status"].value_counts()["test"] == len(test)
    assert not train[EXPERIMENT_LABEL_COLUMN].eq("ambigu").any()
    assert not test[EXPERIMENT_LABEL_COLUMN].eq("ambigu").any()
    excluded_groups = split_dataset.loc[
        split_dataset[GROUP_CONFLICT_COLUMN], GROUP_ID_COLUMN
    ].unique()
    assert not train[GROUP_ID_COLUMN].isin(excluded_groups).any()
    assert not test[GROUP_ID_COLUMN].isin(excluded_groups).any()
