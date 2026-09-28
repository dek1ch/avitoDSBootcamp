import pandas as pd
from bm25_search import (
    CANDIDATE_COUNT,
    ITEM_TEXT_COLUMNS,
    LOCAL_COUNT,
    build_index,
    combine_candidates,
    find_candidates,
    select_top_ids,
)

# Все признаки запроса
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Гиперпараметры
VALIDATION_SIZE = 10_000 # Размер выборки

# Берём 10 000 разных сочетаний признаков запроса вокруг середины train
def main():
    # Читаем train
    choices = pd.read_parquet("train.parquet", columns=QUERY_COLUMNS + ["item_id"])

    # Оставляем только уникальные сочетания признаков запроса
    unique_queries = choices[QUERY_COLUMNS].drop_duplicates()

    # находим середину выборки и её позицию
    middle_row = len(choices) // 2
    middle_position = unique_queries.index.searchsorted(middle_row)

    # Выделяем начало выборки
    start = middle_position - VALIDATION_SIZE // 2
    selected = unique_queries.iloc[start : start + VALIDATION_SIZE]

    query_keys = list(selected.itertuples(index=False, name=None))

    # Проходим по всему train, чтобы собрать словарь из правильного запроса и подходящих под него объявлений
    correct_items = {key: set() for key in query_keys}
    for row in choices.itertuples(index=False, name=None):
        key = row[:-1]
        if key in correct_items:
            correct_items[key].add(row[-1])

    # Составляем выборку запросов, у котоых было 2 или более разных выбранных item_id
    repeated_queries = [(key, correct_items[key]) for key in query_keys if len(correct_items[key]) >= 2]

    # Небольшой лог
    print("Запросов в середине train:", len(query_keys), flush=True)
    print("Из них с разными выбранными объявлениями:", len(repeated_queries), flush=True)
    del choices, unique_queries, correct_items

    # Проверка на заполненность выборки
    if not repeated_queries:
        print("Таких запросов нет")
        return

    # Достаём из train необходимые нам параметры для токенизации объявлений
    raw_items = pd.read_parquet(
        "train.parquet",
        columns=["item_id", "item_location_id"] + ITEM_TEXT_COLUMNS,
    )
    items, index = build_index(raw_items)
    del raw_items

    # ещё один мини лог
    print("Объявлений в поиске:", len(items), flush=True)

    # Для проверки старого и нового поиска с помощью метрики recall
    old_recall_sum = 0.0
    new_recall_sum = 0.0

    # Показатели насколько поиск стал лучше или хуже
    better = 0
    same = 0
    worse = 0

    for number, (key, correct) in enumerate(repeated_queries, start=1):
        # Получаем запрос из query_keys
        query = {
            "search_query": key[0],
            "search_infm_params_text": key[3],
        }
        location_id = key[1]

        # Старый вариант: общий bm25 поиск после бонус за локацию
        global_candidates = find_candidates(query, items, index, n=CANDIDATE_COUNT)
        old_top = select_top_ids(global_candidates, location_id)

        # Новый вариант: к общему списку добавляем местные объявления
        local_candidates = find_candidates(
            query, items, index, n=LOCAL_COUNT, location_id=location_id
        )
        combined = combine_candidates(global_candidates, local_candidates, LOCAL_COUNT)
        new_top = select_top_ids(combined, location_id)

        # Считаем метрику для каждого запроса отдельно и добавляем для глобального результата
        old_score = len(correct.intersection(old_top)) / len(correct)
        new_score = len(correct.intersection(new_top)) / len(correct)
        old_recall_sum += old_score
        new_recall_sum += new_score

        # Обновление показателей
        if new_score > old_score:
            better += 1
        elif new_score < old_score:
            worse += 1
        else:
            same += 1

        # Уведомление о проделаной работе каждые 1000 пройденных запросов
        if number % 1_000 == 0:
            print("Проверено:", number, flush=True)

    # Считаем метрики и выводим результат
    count = len(repeated_queries)
    old_recall = old_recall_sum / count
    new_recall = new_recall_sum / count
    print("Старый Recall@50:", f"{old_recall:.4%}")
    print("Новый Recall@50:", f"{new_recall:.4%}")
    print("Разница, процентных пунктов:", f"{(new_recall - old_recall) * 100:+.2f}")
    print("Запросов лучше / одинаково / хуже:", better, same, worse)



if __name__ == "__main__":
    main()