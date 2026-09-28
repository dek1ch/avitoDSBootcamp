import pandas as pd
from bm25_search import (
    CANDIDATE_COUNT,
    LOCAL_COUNT,
    build_index,
    combine_candidates,
    find_candidates,
    select_top_ids,
)



def main():
    # Читаем данные из benchmark_queries.parquet и benchmark_items.parquet
    queries = pd.read_parquet("benchmark_queries.parquet")
    raw_items = pd.read_parquet("benchmark_items.parquet")
    items, index = build_index(raw_items)
    del raw_items


    valid_item_ids = set(items["item_id"])
    predictions = []


    for number, (_, query) in enumerate(queries.iterrows(), start=1):
        # Ищем общие объявления среди объявлений(items) исходя из запроса(query) (до 1000 кандидатов)
        candidates = find_candidates(query, items, index, n=CANDIDATE_COUNT)

        # Ищем локальные объявления исходя из запроса (до 50 кандидатов)
        local_candidates = find_candidates(
            query, items, index, n=LOCAL_COUNT,
            location_id=query["search_location_id"],
        )

        # Объединяем общие и локальные объявления
        combined = combine_candidates(candidates, local_candidates, LOCAL_COUNT)

        # Добавляем бонус кандидатам из локации запроса и выбираем до 50 объявлений
        item_ids = select_top_ids(combined, query["search_location_id"])

        # Проверка, что каждая строка ответа содержит до 50 существующих item_id без повторов
        assert len(item_ids) <= 50 # Проверка на количество
        assert len(item_ids) == len(set(item_ids)) # Проверка на уникальность
        assert set(item_ids).issubset(valid_item_ids) # Проверка, что значения item_ids есть в valid_item_ids

        # Записываем выбранные item_id одной строкой через пробел
        predictions.append(" ".join(item_ids))

        # Уведомление о проделаной работе каждые 500 пройденных запросов
        if number % 500 == 0:
            print("Обработано запросов:", number, flush=True)

    # Формирование ответа на задание
    answer = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "answer": predictions,
        }
    )

    # Проверка на соответствие набора query_id исходному файлу
    # На уникальность
    assert queries["query_id"].is_unique
    assert answer["query_id"].is_unique
    # На длину айди запроса
    assert answer["query_id"].str.len().eq(16).all()

    answer.to_csv("answer.csv", index=False, encoding="utf-8")
    print("Создан answer.csv. Строк:", len(answer))



if __name__ == "__main__":
    main()