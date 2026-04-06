import json

from app.ingestion.generator import DOC_TYPES, generate_synthetic_dataset


def test_dataset_builder_generates_diverse_doc_types(tmp_path):
    stats = generate_synthetic_dataset(
        data_dir=tmp_path,
        num_properties=40,
        docs_per_property=6,
        eval_queries=60,
        seed=23,
    )
    assert stats["documents"] >= 200

    docs_path = tmp_path / "raw" / "property_documents.jsonl"
    observed_types = set()
    metadata_checks = 0
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            observed_types.add(row["doc_type"])
            md = row["metadata"]
            for field in ["city", "neighborhood", "asking_price", "noi_estimate", "property_tax_annual"]:
                assert field in md
            metadata_checks += 1
    assert metadata_checks > 0
    assert len(observed_types.intersection(set(DOC_TYPES))) >= 8
