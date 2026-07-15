from summarize_candidates import pick_metric, quality_avg


def test_all_dataset_mini_summary_outranks_single_dataset_fit() -> None:
    rows = [
        {
            "dataset": "vrsbench",
            "stage": "fit",
            "accuracy_by_task": {"vrsbench/vqa": {"acc": 2 / 3}},
        },
        {
            "dataset": "all",
            "stage": "mini",
            "accuracy_by_task": {
                "vrsbench/vqa": {"acc": 0.38},
                "xlrs_lite/vqa": {"acc": 0.76},
                "mme_realworld/vqa": {"acc": 0.52},
            },
        },
    ]

    assert pick_metric(rows, "vrsbench", "vqa") == 0.38
    assert pick_metric(rows, "xlrs", "vqa") == 0.76
    assert pick_metric(rows, "mme", "vqa") == 0.52


def test_quality_average_uses_all_three_vqa_benchmarks() -> None:
    assert quality_avg(0.38, 0.76, 0.52) == 0.5533
