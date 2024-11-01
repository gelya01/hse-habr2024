# coding: windows-1251

# # ������ ������ ����� �����
import logging
import requests
from tqdm.asyncio import tqdm_asyncio
from bs4 import BeautifulSoup
import pandas as pd
import time
from random import randint
import asyncio
import nest_asyncio
from typing import List, Dict
import aiohttp
from fake_useragent import UserAgent, FakeUserAgentError

logging.basicConfig(filename='parser.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
nest_asyncio.apply()
pd.options.display.max_colwidth = 90


# ##Random user agent, ���� � �����
# ������� ��� �������� ���������� user_agent
def get_random_user_agent() -> str:
    """
    ���������� ��������� User-Agent ��� ������������� � HTTP-��������

    � ������ ������ ��� ��������� ���������� User-Agent ���������� ����������� User-Agent

    ����������:
        str: ��������� User-Agent ��� ����������� User-Agent (���� ��������� ������)
    """
    try:
        ua = UserAgent()
        return ua.random
    except FakeUserAgentError:
        # ���������� ����������� User-Agent � ������ ������
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'


# ����� ��������� ���� � ���������
cookies = {'hl': 'ru'}
headers = {
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7'
        }

# ## ������ �����


# ������� ��� ��������� ���������� � �����
def parse_habr_hubs():
    """
    ������ ���������� � ����� � ����� ����� � ���������� ������ � ���� DataFrame
    ������� ��������������� �������� �������� �����, �������� ��������, ��������,
    URL-�����, ������� � ���������� ����������� ������� ����

    ����������:
        pd.DataFrame: ������� � ����������� � �����, ��� ������ ������ ������������ ���� ���

    ������� � ������������ ����������:
        - �������� ���� (Title)
        - �������� ���� (Description)
        - URL-����� ���� (URL)
        - ������� ���� (Rating)
        - ���������� ����������� (Subscribers_cnt)
    """
    page_number = 1
    all_titles: List[str] = []
    all_descriptions: List[str] = []
    all_urls: List[str] = []
    all_ratings: List[int] = []
    all_subscribers: List[int] = []

    while True:
        url = f'https://habr.com/ru/hubs/page{page_number}/'
        response = requests.get(url, headers=headers, cookies=cookies)
        if response.status_code != 200:
            print(f"������ ��� ��������� URL: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')

        # ���������, ���� �� �� �������� ����
        hub_elements = soup.find_all('a', class_='tm-hub__title')
        if not hub_elements:
            print("��������� ����� ������ �����")
            break  # ���� ����� ���, ������� �� �����

        # �������� �����
        titles = [title.get_text(strip=True) for title in hub_elements]
        all_titles.extend(titles)

        # ���������� �������� �����
        description_elements = soup.find_all('div',
                                             class_='tm-hub__description')
        descriptions = [descr.get_text(strip=True)
                        for descr in description_elements]
        all_descriptions.extend(descriptions)

        # ������ �� ����
        urls = ['https://habr.com' + url['href'] for url in hub_elements]
        all_urls.extend(urls)

        # �������� �����
        rating_elements = soup.find_all('div',
                                        class_='tm-hubs-list__hub-rating')
        ratings = []
        for rating in rating_elements:
            rating_text = rating.get_text(strip=True).replace('�������', '')
            try:
                ratings.append(int(float(rating_text)))
            except ValueError:
                ratings.append(None)
        all_ratings.extend(ratings)

        # ���������� ����������� ����
        subscriber_elements = soup.find_all('div',
                                            class_='tm-hubs-list__hub-subscribers')
        subscribers = []
        for subscriber in subscriber_elements:
            text = subscriber.get_text(strip=True).replace('����������', '')
            # ����������� ���������� ����������� (� �������) � �����
            if 'K' in text:
                subscribers.append(int(float(
                    text.replace('K', '').replace(',', '.')) * 1000))
            else:
                try:
                    subscribers.append(int(text))
                except ValueError:
                    subscribers.append(None)
        all_subscribers.extend(subscribers)

        print(f"�������� {page_number} ����������")
        page_number += 1
        # ��������� �������� �� 1 �� 3 ������, ����� �� ������� ���� �� �������
        time.sleep(randint(1, 3))

    # �������� DataFrame �� ����� ������
    data = {
        'Title': all_titles,
        'Description': all_descriptions,
        'URL': all_urls,
        'Rating': all_ratings,
        'Subscribers_cnt': all_subscribers}
    df = pd.DataFrame(data)
    return df


# ## ���������� ���������� ������� ������ �����

# ����������� �� 20 ������������� �������� ����� Semaphore
semaphore = asyncio.Semaphore(20)


# ����������� ������� ��� ��������� ���������� ������� ������ �����
async def get_hub_pages_count_async(hub_url: str,
                                    session: aiohttp.ClientSession) -> int:
    """
    ���������� �������� ���������� ������� ������ ���� �� URL
    ���������:
        hub_url (str): URL-����� ����, ������ ����� �������� ���������� �������
        session (aiohttp.ClientSession): ������ ��� ���������� ������������ �������
    ����������:
        int: ����� ���������� ������� � ���� ��� None (���� ��������� ������)
    """
    async with semaphore:
        try:
            async with session.get(hub_url, cookies=cookies) as response:
                if response.status != 200:
                    print(f"������ ��� ��������� �������� ���� {hub_url}: {response.status}")
                    return None
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')

                # ������� ���� ���������
                pagination = soup.find('div', class_='tm-pagination')
                if pagination:
                    pages = pagination.find_all('a',
                                                class_='tm-pagination__page')
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
            print(f"���������� ��� ��������� {hub_url}: {e}")
        return None


# ������� ��� ����� ����������� �� ���� �������
async def process_urls(urls: List[str]) -> List[int]:
    """
    ���������� ������������ ������ URL, �������� ���������� ������� ��� ������� ����

    ���������:
        urls (List[str]): ������ URL-������� ����� ��� ���������

    ����������:
        List[int]: ������, ���������� ���������� ������� ��� ������� ����
                    ��� None (���� ��������� ������)
    """
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [get_hub_pages_count_async(url, session) for url in urls]
        results = await tqdm_asyncio.gather(*tasks, desc='�������� ���������� ������� � �����')
    return results


# ����������� ������� ��������� HTML-��������
async def fetch(session: aiohttp.ClientSession, url: str) -> str:
    """
    ���������� ��������� ������ ��� ��������� HTML-�������� �������� �� URL

    ���������:
        session (aiohttp.ClientSession): ����������� ������ ��� ���������� HTTP-�������
        url (str): URL-����� ��������, ������� ����� ���������

    ����������:
        str: ��������� ������� �������� ��� None (� ������ ������)
    """
    async with semaphore:
        try:
            async with session.get(url, headers=headers, cookies=cookies, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    await asyncio.sleep(randint(1, 3))
                    return content
                else:
                    print(f"������ ��� ��������� {url}: {response.status}")
                    await asyncio.sleep(randint(1, 3))
                    return None
        except Exception as e:
            print(f"���������� ��� ��������� {url}: {e}")
            await asyncio.sleep(randint(1, 3))
            return None


# ����������� ������� �������� ������ �� HTML ��������
async def parse_articles_from_content(content: str,
                                      hub_name: str,
                                      df_full: pd.DataFrame) -> List[Dict[str, str]]:
    """
    ������ ������ �� HTML-��������

    ���������:
        content (str): HTML-������� ��������
        hub_name (str): �������� ����, � �������� ��������� ������
        df_full (pd.DataFrame): DataFrame, ���������� ��������� ������ (��� ���������� ������)

    ����������:
        List[Dict[str, str]]: ������ �������� � ����������� � ������� (���������, URL � ���)
    """
    soup = BeautifulSoup(content, 'html.parser')
    articles = []

    article_elements = soup.find_all('a', class_='tm-title__link')
    for article in article_elements:
        title = article.get_text(strip=True)
        link = 'https://habr.com' + article['href']
        if len(df_full[df_full['URL'] == link]) < 1:  # �������, ����� ��������� ����� ������
            articles.append({'Title': title, 'URL': link, 'Hub': hub_name})

    return articles


# ����������� ������� �������� ������ � ����
async def parse_habr_articles_in_hub(hub_url: str, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    ���������� ������ ������ �� ����, �������� ��� �������� � ������ �� ������

    ���������:
        hub_url (str): URL-����� ����, �� �������� ����������� ������
        df_full (pd.DataFrame): DataFrame, ���������� ��������� ������ (��� ���������� ������)

    ����������:
        pd.DataFrame: DataFrame �� �������� �� ����, ��� ���� �������� ������, URL � �������� ����
    """
    all_articles = []

    async with aiohttp.ClientSession() as session:
        # ��������� ������� �������� ����
        response = await fetch(session, hub_url)
        if not response:
            print(f"�� ������� �������� ������� �������� ����: {hub_url}")
            return None

        soup = BeautifulSoup(response, 'html.parser')

        # ���������� �������� ����
        hub_name_element = soup.find(
            'h1', class_='tm-hub-card__name tm-hub-card__name_variant-base tm-hub-card__name')
        if hub_name_element:
            hub_name = hub_name_element.get_text(strip=True)
        else:
            hub_name = 'Unknown'

        print(f"������� ����: {hub_name}")

        # ������� ����� ���������� �������
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

        # ������� ������ ��� ���� �������
        tasks = []
        for page in range(1, total_pages + 1):
            url = f"{hub_url}page{page}/"
            tasks.append(fetch(session, url))

        # ��������� ������ � �������� ����������
        responses = await tqdm_asyncio.gather(*tasks, desc='�������� �������')

        # ������ �������� ������ ��������
        for content in responses:
            if content:
                articles = await parse_articles_from_content(content, hub_name, df_full)
                all_articles.extend(articles)

    # �������� DataFrame �� ����� �������� �� ������
    df = pd.DataFrame(all_articles)
    return df


# ## ������ ������

# ����������� ������ ������
async def parse_habr_article(
        url: str, session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore, counter: List[int],
        lock: asyncio.Lock) -> pd.DataFrame:

    """
    ���������� ������ ������ � ����� �� URL, �������� ������ ��������

    ���������:
        url (str): URL-����� ������ ��� ��������
        session (aiohttp.ClientSession): ����������� ������ ��� ���������� HTTP-�������
        semaphore (asyncio.Semaphore): ����������� �� ������������� �������
        counter (List[int]): ������� ������������ ������
        lock (asyncio.Lock): ���������� ��� ����������� ���������� ��������

    ����������:
        pd.DataFrame: DataFrame � ����������� � ������ (���������, �����, ���� � �.�.)
            ��� None � ������ ������
    """
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    await asyncio.sleep(randint(1, 3))
                else:
                    logging.warning(f"������ {response.status} ��� ��������� URL: {url}")
                    await asyncio.sleep(randint(1, 3))
                    return None
        except Exception as e:
            logging.exception(f"���������� ��� ��������� {url}: {e}")
            await asyncio.sleep(randint(1, 3))
            return None

    try:
        soup = BeautifulSoup(content, 'html.parser')

        # �������� ������
        title_text = soup.find('h1', class_='tm-title tm-title_h1')
        title = title_text.get_text(strip=True) if title_text else None

        # ��� ������
        author_text = soup.find('a', class_='tm-user-info__username')
        author = author_text.get_text(strip=True) if author_text else None

        # ���� ����������
        date_text = soup.find('time')
        pub_date = pd.to_datetime(date_text['datetime']) if date_text else None

        # ����
        hub_elements = soup.find_all('a', class_='tm-hubs-list__link')
        hubs = [hub.get_text(strip=True) for hub in hub_elements]

        # ������ �������� ��� ���. ����
        individual_or_company = ('company' if any('���� ��������' in x for x in hubs)
                                 else 'individual')

        # ����
        tag_elements = soup.find_all('a', class_='tm-tags-list__link')
        tags = [tag.get_text(strip=True) for tag in tag_elements]

        # ���������� ������
        content_text = soup.find('div', class_='tm-article-body')
        content = content_text.get_text(separator='\n', strip=True) if content_text else None

        # ���������� ������������
        comments_text = soup.find('span', class_='tm-article-comments-counter-link__value')
        comments = int(comments_text.get_text(strip=True) if comments_text else 0)

        # ���������� ����������
        views_text_ = soup.find('span', class_='tm-icon-counter__value')
        views_text = views_text_.get_text(strip=True) if views_text_ else '0'
        try:
            if 'K' in views_text:
                views = int(float(views_text.replace('K', '').replace(',', '.')) * 1000)
            else:
                views = int(views_text)
        except ValueError:
            views = -1

        # ����� ��������� � �������
        reading_time_text = soup.find('span', class_='tm-article-reading-time__label')
        reading_time = (int(reading_time_text.get_text(strip=True).split()[0])
                        if reading_time_text else None)

        # ���������� ���������� � ��������
        bookmarks_text = soup.find('span', class_='bookmarks-button__counter')
        bookmark = int(bookmarks_text.get_text(strip=True)) if bookmarks_text else None

        # ������ �� ��������
        images = content_text.find_all("img") if content_text else []
        images_links = [img['src'] for img in images if img.has_attr('src')]

        # ������� ������
        article_rating_tag = soup.find('span', class_='tm-votes-meter__value')
        if article_rating_tag:
            article_rating = article_rating_tag.get_text(strip=True)
        else:
            article_rating_tag = soup.find('span', class_='tm-votes-lever__score-counter')
            article_rating = article_rating_tag.get_text(strip=True) if article_rating_tag else '0'

        # ���������� ��� ���������� ������� ������
        article_rating = article_rating or '0'
        positive_negative = 'negative' if '-' in article_rating else 'positive'
        article_rating_value = (int(float(article_rating.replace('+', '').replace('-', '')))
                                if article_rating else 0)

        # �������� ��������� DataFrame �� ��������
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

    # ���������� ���������� �������� �� ������ �������
        async with lock:
            counter[0] += 1

        return df
    except Exception as e:
        logging.exception(f"������ ��� �������� �������� {url}: {e}")
        return None


# ������� ��� ���������� ������ �� n ������ � ����������� �������������� ���������
def split_list(lst: List, n: int) -> List[List]:
    """
    ��������� ������ �� n ���������� � ����������� �������������� ��������� ������

    ���������:
        lst (List): �������� ������ ��� ����������
        n (int): ���������� ���������� ��� ����������

    ����������:
        List[List]: ������ �� n ����������, ���������� �������� ��������� ������
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# ����������� ������� ��������� ������
async def parse_article(
    urls: List[str],
    counter: List[int],
    lock: asyncio.Lock,
    semaphore_num: int = 20
) -> pd.DataFrame:
    """
    ���������� �������� � ������ ������ �� ������ URL

    ���������:
        urls (List[str]): ������ URL-������� ������ ��� ��������
        counter (List[int]): ������� ������������ ������
        lock (asyncio.Lock): ���������� ��� ����������� ���������� ��������
        semaphore_num (int): ������������ ���������� ������������� �������� (�� ��������� 20)

    ����������:
        pd.DataFrame: DataFrame, ���������� ������ �� ���� �������� ��������� ������
    """
    # ����������� �� 20 ������������� ��������(����� �������� ��� ������ �� ��������)
    semaphore = asyncio.Semaphore(semaphore_num)
    async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
        tasks = [parse_habr_article(url, session, semaphore, counter, lock) for url in urls]
        results = []
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await future
            results.append(result)
        dfs = [df for df in results if df is not None]
        final_df = pd.concat(dfs, ignore_index=True)
        # ������� ���������� ������������ ������
        print(f"���������� {len(final_df)} ������")
        return final_df


# ����������� ��������� ������ ������� �� �������� (hubs_parts)
async def process_part(
    urls_chunk: List[str],
    part_number: int,
    counter: List[int],
    lock: asyncio.Lock
) -> None:
    """
    ���������� ������������ ����� URL ��� �������� ������, ��������� ���������� � ����

    ���������:
        urls_chunk (List[str]): ����� ������ URL-������� ��� ��������
        part_number (int): ����� ������� ����� (������������ ��� ����� �����)
        counter (List[int]): ������� ������������ ������
        lock (asyncio.Lock): ���������� ��� ����������� ���������� ��������

    ����������:
        None: ������� ��������� ��������� � ���� � ������ �� ����������
    """
    # ����������� �� 50 ������������� �������� (��� ��������� �������� ���� ����������� ������,
    # ���������� ������ ���������� ��������)
    final_df = await parse_article(urls_chunk, counter, lock, semaphore_num=50)
    if final_df is not None:
        filename = f'articles_part_{part_number}.parquet'
        final_df.to_parquet(filename, index=False)
        print(f"����� {part_number} ��������� � ���� {filename}")
    else:
        print(f"��� ������ ��� ���������� � ����� {part_number}")
