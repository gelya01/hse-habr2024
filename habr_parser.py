#!/usr/bin/env python
# coding: utf-8

# # Парсер статей Хабра Хабра

# In[1]:


# Импорты
import logging
logging.basicConfig(
    filename='parser.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import requests
from tqdm.asyncio import tqdm_asyncio
from bs4 import BeautifulSoup
import pandas as pd
import time
from random import randint
import asyncio
import nest_asyncio
nest_asyncio.apply()
import aiohttp
from fake_useragent import UserAgent, FakeUserAgentError
pd.options.display.max_colwidth = 90


# ## Random user agent, куки и хэдэр

# In[2]:


# Функция для создания случайного user_agent
def get_random_user_agent():
    try:
        ua = UserAgent()
        return ua.random
    except FakeUserAgentError:
        # Используем стандартный User-Agent в случае ошибки
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

# Задаём куки и заголовок
cookies = {'hl': 'ru'}
headers = {
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
        }


# ## Парсер хабов

# In[3]:


# Функция для получения информации о хабах 
def parse_habr_hubs():
    page_number = 1
    all_titles = []
    all_descriptions = []
    all_urls = []
    all_ratings = []
    all_subscribers = []

    while True:
        url = f'https://habr.com/ru/hubs/page{page_number}/'
        response = requests.get(url, headers=headers, cookies=cookies)
        if response.status_code != 200:
            print(f"Ошибка при получении URL: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')

        # Проверяем, есть ли на странице хабы
        hub_elements = soup.find_all('a', class_='tm-hub__title')
        if not hub_elements:
            print("Достигнут конец списка хабов")
            break  # Если хабов нет, выходим из цикла

        # Названия хабов
        titles = [title.get_text(strip=True) for title in hub_elements]
        all_titles.extend(titles)

        # Извлечение описания хабов
        description_elements = soup.find_all('div', class_='tm-hub__description')
        descriptions = [descr.get_text(strip=True) for descr in description_elements]
        all_descriptions.extend(descriptions)

        # Ссылки на хабы
        urls = ['https://habr.com' + url['href'] for url in hub_elements]
        all_urls.extend(urls)

        # Рейтинги хабов
        rating_elements = soup.find_all('div', class_='tm-hubs-list__hub-rating')
        ratings = []
        for rating in rating_elements:
            rating_text = rating.get_text(strip=True).replace('Рейтинг','')
            try:
                ratings.append(int(float(rating_text)))
            except ValueError:
                ratings.append(None) 
        all_ratings.extend(ratings)

        # Количество подписчиков хаба
        subscriber_elements = soup.find_all('div', class_='tm-hubs-list__hub-subscribers')
        subscribers = []
        for subscriber in subscriber_elements:
            text = subscriber.get_text(strip=True).replace('Подписчики','')
            # Преобразуем количество подписчиков (в тысячах) в число
            if 'K' in text:
                subscribers.append(int(float(text.replace('K', '').replace(',', '.')) * 1000))
            else:
                try:
                    subscribers.append(int(text))
                except ValueError:
                    subscribers.append(None)  
        all_subscribers.extend(subscribers)

        print(f"Страница {page_number} обработана")
        page_number += 1
        # Случайная задержка от 1 до 3 секунд, чтобы не словить блок от сервера
        time.sleep(randint(1, 3))

    # Создание DataFrame со всеми хабами
    data = {
        'Title': all_titles,
        'Description': all_descriptions,
        'URL': all_urls,
        'Rating': all_ratings,
        'Subscribers_cnt': all_subscribers
    }

    df = pd.DataFrame(data)
    return df


# In[ ]:


hubs = parse_habr_hubs()


# ## Добавление количества страниц внутри хабов

# In[5]:


# Ограничение на 20 одновременных запросов через Semaphore
semaphore = asyncio.Semaphore(20)
# Асинхронная функция для получения количества страниц внутри хабов
async def get_hub_pages_count_async(hub_url, session):
    async with semaphore:
        try:
            async with session.get(hub_url, cookies=cookies) as response:
                if response.status != 200:
                    print(f"Ошибка при получении страницы хаба {hub_url}: {response.status}")
                    return None
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')

                # Находим блок пагинации
                pagination = soup.find('div', class_='tm-pagination')
                if pagination:
                    pages = pagination.find_all('a', class_='tm-pagination__page')
                    if pages:
                        last_page = pages[-1].get_text(strip=True)
                        try:
                            total_pages = int(last_page)
                        except ValueError:
                            total_pages = 1
                    else:
                        total_pages = 1
                else:
                    total_pages = 1
                return total_pages
        except aiohttp.ClientError as e:
            print(f"Исключение при получении {hub_url}: {e}")
        return None

# Функция для сбора результатов со всех страниц
async def process_urls(urls):
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [get_hub_pages_count_async(url, session) for url in urls]
        results = await tqdm_asyncio.gather(*tasks, desc='Получаем количество страниц в хабах')
    return results


# In[ ]:


url_lst = hubs['URL'].tolist() # список всех url
res = asyncio.run(process_urls(url_lst)) # получаем количество страниц
hubs.insert(5, 'Pages_cnt', res) # добавляем количество страниц в датафрейм хабов 


# In[ ]:


hubs


# In[23]:


# Cохраняем датафрейм со ссылками на хабы
hubs.to_excel('hubs_urls.xlsx', index_label='ID')


# ## Парсер ссылок на статьи внутри хабов

# In[8]:


df_full = pd.DataFrame(columns=['Title', 'URL', 'Hub']) # Создание итогового DataFrame
semaphore = asyncio.Semaphore(20)  # Максимум 20 одновременных запросов

# Асинхронная функция получения url статьи
async def fetch(session, url):
    async with semaphore:
        try:
            async with session.get(url, headers=headers, cookies=cookies, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    await asyncio.sleep(randint(1, 3))
                    return content
                else:
                    print(f"Ошибка при получении {url}: {response.status}")
                    await asyncio.sleep(randint(1, 3))
                    return None
        except Exception as e:
            print(f"Исключение при получении {url}: {e}")
            await asyncio.sleep(randint(1, 3))
            return None

# Асинхронная функция парсинга статьи из HTML контента
async def parse_articles_from_content(content, hub_name, df_full):
    soup = BeautifulSoup(content, 'html.parser')
    articles = []

    article_elements = soup.find_all('a', class_='tm-title__link')
    for article in article_elements:
        title = article.get_text(strip=True)
        link = 'https://habr.com' + article['href']
        if len(df_full[df_full['URL']==link]) < 1: #условие, чтобы исключить дубли статей
            articles.append({'Title': title, 'URL': link, 'Hub': hub_name})

    return articles

# Асинхронная функция парсинга статей в хабе
async def parse_habr_articles_in_hub(hub_url, df_full):
    all_articles = []

    async with aiohttp.ClientSession() as session:
        # Получение главной страницы хаба
        response = await fetch(session, hub_url)
        if not response:
            print(f"Не удалось получить главную страницу хаба: {hub_url}")
            return None

        soup = BeautifulSoup(response, 'html.parser')

        # Извлечение названия хаба
        hub_name_element = soup.find('h1', class_='tm-hub-card__name tm-hub-card__name_variant-base tm-hub-card__name')
        if hub_name_element:
            hub_name = hub_name_element.get_text(strip=True)
        else:
            hub_name = 'Unknown'

        print(f"Парсинг хаба: {hub_name}")

        # Находим общее количество страниц
        pagination = soup.find('div', class_='tm-pagination')
        if pagination:
            pages = pagination.find_all('a', class_='tm-pagination__page')
            if pages:
                last_page = pages[-1].get_text(strip=True)
                try:
                    total_pages = int(last_page)
                except ValueError:
                    total_pages = 1
            else:
                total_pages = 1
        else:
            total_pages = 1

        # Создаем задачи для всех страниц
        tasks = []
        for page in range(1, total_pages + 1):
            url = f"{hub_url}page{page}/"
            tasks.append(fetch(session, url))

        # Запускаем задачи и собираем результаты
        responses = await tqdm_asyncio.gather(*tasks, desc='Загрузка страниц')

        # Парсим контента каждой страницы
        for content in responses:
            if content:
                articles = await parse_articles_from_content(content, hub_name, df_full)
                all_articles.extend(articles)

    # Создание DataFrame со всеми ссылками на статьи
    df = pd.DataFrame(all_articles)
    return df


# In[ ]:


# Запуск асинхронной функции
for i in range(len(hubs)):
    hub_url = hubs.iloc[i]['URL'] + 'articles/'
    df = asyncio.run(parse_habr_articles_in_hub(hub_url, df_full))
    df_full = pd.concat([df_full, df], ignore_index=True)


# In[ ]:


if df_full is not None:
    print(f"Всего собрано статей: {len(df_full)}")
    display(df_full.head())


# In[ ]:


#убираем дубликаты (если они где-то пробрались)
display(df_full['URL'].duplicated().sum())
hubs_full = df_full.drop_duplicates(subset='URL')


# In[ ]:


hubs_full


# In[ ]:


# Сохраняем итоговый датафрейм со ссылками на все статьи
hubs_full.to_parquet('hubs_to_articles_urls.parquet', index=False)


# ## Парсер статей

# In[14]:


# Асинхронный парсер статей
async def parse_habr_article(url, session, semaphore, counter, lock):
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    await asyncio.sleep(randint(1, 3))
                else:
                    logging.warning(f"Ошибка {response.status} при получении URL: {url}")
                    await asyncio.sleep(randint(1, 3))
                    return None
        except Exception as e:
            logging.exception(f"Исключение при получении {url}: {e}")
            await asyncio.sleep(randint(1, 3))
            return None

    try:
        soup = BeautifulSoup(content, 'html.parser')

        # Название статьи
        title_text = soup.find('h1', class_='tm-title tm-title_h1')
        title = title_text.get_text(strip=True) if title_text else None

        # Имя автора
        author_text = soup.find('a', class_='tm-user-info__username')
        author = author_text.get_text(strip=True) if author_text else None

        # Дата публикации
        date_text = soup.find('time')
        pub_date = pd.to_datetime(date_text['datetime']) if date_text else None

        # Хабы
        hub_elements = soup.find_all('a', class_='tm-hubs-list__link')
        hubs = [hub.get_text(strip=True) for hub in hub_elements]

        # Статья компании или физ. лица
        individual_or_company = 'company' if any('Блог компании' in x for x in hubs) else 'individual'

        # Теги
        tag_elements = soup.find_all('a', class_='tm-tags-list__link')
        tags = [tag.get_text(strip=True) for tag in tag_elements]

        # Содержимое статьи
        content_text = soup.find('div', class_='tm-article-body')
        content = content_text.get_text(separator='\n', strip=True) if content_text else None

        # Количество комментариев
        comments_text = soup.find('span', class_='tm-article-comments-counter-link__value')
        comments = int(comments_text.get_text(strip=True) if comments_text else 0)

        # Количество просмотров
        views_text_ = soup.find('span', class_='tm-icon-counter__value')
        views_text = views_text_.get_text(strip=True) if views_text_ else '0'
        try:
            if 'K' in views_text:
                views = int(float(views_text.replace('K', '').replace(',', '.')) * 1000)
            else:
                views = int(views_text)
        except ValueError:
            views = -1  

        # Время прочтения в минутах
        reading_time_text = soup.find('span', class_='tm-article-reading-time__label')
        reading_time = int(reading_time_text.get_text(strip=True).split()[0]) if reading_time_text else None

        # Количество добавлений в закладки
        bookmarks_text = soup.find('span', class_='bookmarks-button__counter')
        bookmark = int(bookmarks_text.get_text(strip=True)) if bookmarks_text else None

        # Ссылки на картинки
        images = content_text.find_all("img") if content_text else []
        images_links = [img['src'] for img in images if img.has_attr('src')]

        # Рейтинг статьи
        article_rating_tag = soup.find('span', class_='tm-votes-meter__value')
        if article_rating_tag:
            article_rating = article_rating_tag.get_text(strip=True)
        else:
            article_rating_tag = soup.find('span', class_='tm-votes-lever__score-counter')
            article_rating = article_rating_tag.get_text(strip=True) if article_rating_tag else '0'

        # Позитивный или негативный рейтинг статьи
        article_rating = article_rating or '0'
        positive_negative = 'negative' if '-' in article_rating else 'positive'
        article_rating_value = int(float(article_rating.replace('+', '').replace('-', ''))) if article_rating else 0

        # Создание итогового DataFrame со статьями
        data = {
            'Title': [title],
            'Author': [author],
            'Publication_date': [pub_date],
            'Hubs': [', '.join(hubs)],
            'Tags': [', '.join(tags)],
            'Content': [content],
            'Comments': [comments],
            'Views': [views],
            'URL': [url],
            'Reading_time': [reading_time],
            'Images_links': [', '.join(images_links)],
            'Individ/Company': [individual_or_company],
            'Rating': [article_rating_value],
            'Positive/Negative': [positive_negative],
            'Bookmarks_cnt': [bookmark]
        }
        
        df = pd.DataFrame(data)

    # Безопасное обновление счётчика из разных корутин
        async with lock:
            counter[0] += 1

        return df
    
    except Exception as e:
        logging.exception(f"Ошибка при парсинге страницы {url}: {e}")
        return None


# In[15]:


# Разобъём на 5 частей (для более лёгкой обработки общего массива данных)
def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

urls = hubs_full['URL']
hubs_parts = split_list(urls, 5)


# In[16]:


# Асинхронная функция получения статей
async def parse_article(urls, counter, lock, semaphore_num=20):
    semaphore = asyncio.Semaphore(semaphore_num)  # Ограничение до 20 одновременных запросов(чтобы получить все статьи на странице)
    async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
        tasks = [parse_habr_article(url, session, semaphore, counter, lock) for url in urls]
        results = []
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await future
            results.append(result)
        dfs = [df for df in results if df is not None]
        final_df = pd.concat(dfs, ignore_index=True)
        # Выводим количество обработанных статей
        print(f"Обработано {len(final_df)} статей")
        return final_df

# Асинхронная обработка частей массива со статьями (hubs_parts)
async def process_part(urls_chunk, part_number, counter, lock):
    final_df = await parse_article(urls_chunk, counter, lock, semaphore_num=50) # Ограничение до 50 одновременных запросов (для ускорения процесса берём ограничение больше, потерянные статьи обработаем отдельно)
    if final_df is not None:
        filename = f'articles_part_{part_number}.parquet'
        final_df.to_parquet(filename, index=False)
        print(f"Часть {part_number} сохранена в файл {filename}")
    else:
        print(f"Нет данных для сохранения в части {part_number}")

# Запускаем общий счётчик и блокировку
global_counter = [0]
global_lock = asyncio.Lock()


# In[ ]:


# Запускаем функцию обработки статей
for i, urls_chunk in enumerate(hubs_parts, 1):
    print(f"Начинается обработка части {i} из {len(hubs_parts)}")
    asyncio.run(process_part(urls_chunk, i, global_counter, global_lock))


# In[29]:


# Создаём итоговый датафрейм из 5 отдельных файлов
fin_df = pd.DataFrame()
for part in range(1, len(hubs_parts)+1):
    df_part = pd.read_parquet(f'articles_part_{part}.parquet')
    fin_df = pd.concat([fin_df, df_part], ignore_index=True)


# In[ ]:


fin_df.info()


# In[ ]:


# Находим потерянные при получении статьи (где был Semaphore = 50) URL, сохраняем их и добавляем в итоговый датафрейм
url_dif = list(set(urls) - set(fin_df['URL']))
missed_articles = asyncio.run(parse_article(url_dif, global_counter, global_lock))
missed_articles.to_parquet('missed_articles.parquet', index=False)
fin_df = pd.concat([fin_df, missed_articles], ignore_index=True)


# In[ ]:


# Сохранение итогов
fin_df.to_parquet('habr_articles_parsed_final.parquet', index=False)

