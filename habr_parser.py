# coding: windows-1251
import pandas as pd
import asyncio
import utils

# получаем список всех хабов
hubs = utils.parse_habr_hubs()
url_lst = hubs['URL'].tolist()  # список всех url

# добавление количества страниц
res = asyncio.run(utils.process_urls(url_lst))  # получаем количество страниц
hubs.insert(5, 'Pages_cnt', res)  # добавляем количество страниц в датафрейм хабов

# Cохраняем датафрейм со ссылками на хабы
hubs.to_excel('hubs_urls.xlsx', index_label='ID')

# ## Парсер ссылок на статьи внутри хабов

df_full = pd.DataFrame(columns=['Title', 'URL', 'Hub'])  # Создание итогового DataFrame с хабами

# Запуск асинхронной функции сбора статей внутри хабов
for i in range(len(hubs)):
    hub_url = hubs.iloc[i]['URL'] + 'articles/'
    df = asyncio.run(utils.parse_habr_articles_in_hub(hub_url, df_full))
    df_full = pd.concat([df_full, df], ignore_index=True)

# Смотрим сколько всего статей получилось собрать
if df_full is not None:
    print(f"Всего собрано статей: {len(df_full)}")


# Убираем дубликаты (если они где-то пробрались)
hubs_full = df_full.drop_duplicates(subset='URL')

# Сохраняем итоговый датафрейм со ссылками на все статьи
hubs_full.to_parquet('hubs_to_articles_urls.parquet', index=False)

# Разобъём на 5 частей (для более лёгкой обработки общего массива данных)
urls = hubs_full['URL']
hubs_parts = utils.split_list(urls, 5)

# Запускаем общий счётчик и блокировку
global_counter = [0]
global_lock = asyncio.Lock()


# Запускаем функцию обработки статей
for i, urls_chunk in enumerate(hubs_parts, 1):
    print(f"Начинается обработка части {i} из {len(hubs_parts)}")
    asyncio.run(utils.process_part(urls_chunk, i, global_counter, global_lock))


# Создаём итоговый датафрейм из 5 отдельных файлов
fin_df = pd.DataFrame()
for part in range(1, len(hubs_parts)+1):
    df_part = pd.read_parquet(f'articles_part_{part}.parquet')
    fin_df = pd.concat([fin_df, df_part], ignore_index=True)


fin_df.info()

# Находим потерянные при получении статьи (где был Semaphore = 50) URL
# Сохраняем их и добавляем в итоговый датафрейм
url_dif = list(set(urls) - set(fin_df['URL']))
missed_articles = asyncio.run(utils.parse_article(url_dif, global_counter, global_lock))
missed_articles.to_parquet('missed_articles.parquet', index=False)
fin_df = pd.concat([fin_df, missed_articles], ignore_index=True)

# Сохранение итогов
fin_df.to_parquet('habr_articles_parsed_final.parquet', index=False)
