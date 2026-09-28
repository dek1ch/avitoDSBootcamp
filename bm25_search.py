import bm25s
import numpy as np
import pandas as pd

# Параметры из объявления
ITEM_TEXT_COLUMNS = [
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
]

# Параметры из запроса
QUERY_TEXT_COLUMNS = [
    "search_query",
    "search_infm_params_text",
]

# Гиперпараметры
CANDIDATE_COUNT = 1_000 # Количество вероятных совпадений
LOCAL_COUNT = 50 # Количество объявлений, которые ищутся по локации
LOCATION_BONUS = 12.0 # Бонус за совпадение параметра item_location_id и search_location_id



def clean_text(value):
    """Функция подготовки текста под один шаблон"""

    if pd.isna(value):
        return ""
    
    return str(value).casefold().replace("ё", "е")



def build_index(items):
    """Функция построения BM25-индекса по объявлениям"""

    # В поиске каждый item_id должен встречаться один раз
    items = items.drop_duplicates(subset="item_id").reset_index(drop=True)

    # Объединяем три текстовых поля каждого объявления
    item_texts = (
        items[ITEM_TEXT_COLUMNS[0]].fillna("").astype(str)
        + " "
        + items[ITEM_TEXT_COLUMNS[1]].fillna("").astype(str)
        + " "
        + items[ITEM_TEXT_COLUMNS[2]].fillna("").astype(str)
    )
    item_texts = [clean_text(text) for text in item_texts]

    # Токенизация текста
    tokens = bm25s.tokenize(item_texts, stopwords=None, show_progress=False)
    index = bm25s.BM25()
    index.index(tokens)

    return items, index



def find_candidates(query, items, index, n=CANDIDATE_COUNT, location_id=None):
    """Вернуть кандидатов из всего корпуса или только из одной локации"""

    # Формируем текст запроса
    query_text = " ".join(
        clean_text(query[column]) for column in QUERY_TEXT_COLUMNS
    )

    # Токенизируем запрос
    query_tokens = bm25s.tokenize(
        query_text,
        stopwords=None,
        show_progress=False,
    )

    # Если указан location_id, то обнуляем оценки объявлений из других локаций при поиске,
    # иначе просто применяем bm25 ко всем подобранным объявлениям
    location_mask = None
    if location_id is not None:
        location_mask = items["item_location_id"].to_numpy() == location_id
        n = min(n, int(location_mask.sum()))
    else:
        n = min(n, len(items))

    # Проверка, что у нас есть хотя бы одно объявление
    if n == 0:
        empty = items.iloc[:0].copy()
        empty["bm25_score"] = np.array([], dtype=float)
        return empty

    # Выполняем поиск подходящих под запрос объявлений исходя из index и query_tokens, маска обнуляет оценки других локаций
    positions, scores = index.retrieve(
        query_tokens,
        k=n,
        show_progress=False,
        weight_mask=location_mask,
    )

    # Так как bm25 возвращает номера позиций, то по ним возьмём настоящие объявления
    candidates = items.iloc[positions[0]].copy()
    candidates["bm25_score"] = scores[0]

    # Проверка на случай, если в объявления по локации попали те, у которых параметр bm25_score равен 0 и локация не совпадают
    if location_id is not None:
        candidates = candidates[
            (candidates["item_location_id"] == location_id)
            & (candidates["bm25_score"] > 0)
        ]
    
    return candidates



def combine_candidates(global_candidates, local_candidates, local_count):
    """Добавить к общему списку подходящих объявлений местные объявления исключая повторы"""
    
    columns = ["item_id", "item_location_id", "bm25_score"]
    return pd.concat(
        [global_candidates[columns], local_candidates[columns].head(local_count)],
        ignore_index=True,
    ).drop_duplicates(subset="item_id")



def select_top_ids(candidates, location_id, bonus=LOCATION_BONUS, n=50):
    """Выбрать объявления по bm25 с бонусом за совпадение локации"""

    same_location = candidates["item_location_id"].to_numpy() == location_id
    scores = candidates["bm25_score"].to_numpy() + bonus * same_location
    positions = np.argsort(-scores, kind="stable")[:n]

    return candidates.iloc[positions]["item_id"].tolist()



# Простой пример запуска поиска
if __name__ == "__main__":
    items = pd.read_parquet("benchmark_items.parquet")
    queries = pd.read_parquet("benchmark_queries.parquet")

    items, index = build_index(items)
    query = queries.iloc[0]
    candidates = find_candidates(query, items, index)

    print("Запрос:", query["search_query"])
    print("Фильтры:", query["search_infm_params_text"])
    print(candidates[["item_id", "item_title_raw", "bm25_score"]].head(10).to_string(index=False))
    print("Всего кандидатов:", len(candidates))
