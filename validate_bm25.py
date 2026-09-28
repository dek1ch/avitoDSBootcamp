import random
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

# Все параметры запроса
QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

# Гиперпараметры
VALIDATION_SIZE = 10_000 # Выборка для проверки
RANDOM_SEED = 42 # Зерно генерации



def main():
    # Читаем полностью train выборку
    choices = pd.read_parquet("train.parquet", columns=QUERY_COLUMNS + ["item_id"])

    # Для каждого сочетания признаков запроса сохраняем выбранные item_id
    answers = {}
    for row in choices.itertuples(index=False):
        query_key = (
            row.search_query,
            row.search_location_id,
            row.search_is_delivery_search,
            row.search_infm_params_text,
            row.search_category,
        )
        answers.setdefault(query_key, set()).add(row.item_id)

    # Делаем случайную выборку для проверки bm25
    selected_keys = random.Random(RANDOM_SEED).sample(list(answers), k=VALIDATION_SIZE)
    cases = [(key, answers[key]) for key in selected_keys]
    del choices, answers

    # выбирем объявления, что использовались train для проверки bm25 и токенизируем их
    raw_items = pd.read_parquet("train.parquet", columns=["item_id", "item_location_id"] + ITEM_TEXT_COLUMNS)
    items, index = build_index(raw_items)
    del raw_items

    old_recall = 0.0
    new_recall = 0.0
    better = 0
    same = 0
    worse = 0

    for number, (query_key, correct_ids) in enumerate(cases, start=1):
        # Создаём запрос для текстового поиска и берём его локацию
        query = {
            "search_query": query_key[0],
            "search_infm_params_text": query_key[3],
        }
        location_id = query_key[1]

        # Старый вариант: общий bm25 поиск после бонус за локацию
        global_candidates = find_candidates(query, items, index, n=CANDIDATE_COUNT)
        old_top = select_top_ids(global_candidates, location_id)

        # Новый вариант: к общему списку добавляем местные объявления
        local_candidates = find_candidates(query, items, index, n=LOCAL_COUNT, location_id=location_id)
        combined = combine_candidates(global_candidates, local_candidates, LOCAL_COUNT)
        new_top = select_top_ids(combined, location_id)

        # Считаем метрику для каждого запроса отдельно и добавляем для глобального результата
        old_score = len(correct_ids.intersection(old_top)) / len(correct_ids)
        new_score = len(correct_ids.intersection(new_top)) / len(correct_ids)
        old_recall += old_score
        new_recall += new_score

        # Обновление показателей
        if new_score > old_score:
            better += 1
        elif new_score < old_score:
            worse += 1
        else:
            same += 1

        # Уведомление о проделаной работе каждые 1000 пройденных запросов
        if number % 1_000 == 0:
            print("Проверено запросов:", number, flush=True)

    # Считаем метрики и выводим результат
    old_recall /= VALIDATION_SIZE
    new_recall /= VALIDATION_SIZE
    print("Старый Recall@50 на всех запросах:", f"{old_recall:.4%}")
    print("Новый Recall@50 на всех запросах:", f"{new_recall:.4%}")
    print("Разница, процентных пунктов:", f"{(new_recall - old_recall) * 100:+.2f}")
    print("Запросов лучше / одинаково / хуже:", better, same, worse)



if __name__ == "__main__":
    main()
