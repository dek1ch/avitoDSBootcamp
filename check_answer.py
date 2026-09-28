import re
import pandas as pd



def main():
    # Читаем данные
    queries = pd.read_parquet("benchmark_queries.parquet", columns=["query_id"])
    items = pd.read_parquet("benchmark_items.parquet", columns=["item_id"])
    answer = pd.read_csv("answer.csv", dtype=str, keep_default_na=False)

    # Делаем дополнительные проверки
    assert answer.columns.tolist() == ["query_id", "answer"]
    assert len(answer) == len(queries)
    assert set(answer["query_id"]) == set(queries["query_id"])

    # Проверка каждой записи
    valid_item_ids = set(items["item_id"])
    for query_id, text in answer.itertuples(index=False, name=None):
        item_ids = text.split() if text else []
        assert len(item_ids) <= 50, query_id
        assert len(item_ids) == len(set(item_ids)), query_id
        assert all(re.fullmatch(r"[0-9a-f]{16}", item_id) for item_id in item_ids), query_id
        assert set(item_ids).issubset(valid_item_ids), query_id

    print("Формат answer.csv верен. Запросов:", len(answer))



if __name__ == "__main__":
    main()