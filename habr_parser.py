# coding: windows-1251
import pandas as pd
import asyncio
import utils

# �������� ������ ���� �����
hubs = utils.parse_habr_hubs()
url_lst = hubs['URL'].tolist()  # ������ ���� url

# ���������� ���������� �������
res = asyncio.run(utils.process_urls(url_lst))  # �������� ���������� �������
hubs.insert(5, 'Pages_cnt', res)  # ��������� ���������� ������� � ��������� �����

# C�������� ��������� �� �������� �� ����
hubs.to_excel('hubs_urls.xlsx', index_label='ID')

# ## ������ ������ �� ������ ������ �����

df_full = pd.DataFrame(columns=['Title', 'URL', 'Hub'])  # �������� ��������� DataFrame � ������

# ������ ����������� ������� ����� ������ ������ �����
for i in range(len(hubs)):
    hub_url = hubs.iloc[i]['URL'] + 'articles/'
    df = asyncio.run(utils.parse_habr_articles_in_hub(hub_url, df_full))
    df_full = pd.concat([df_full, df], ignore_index=True)

# ������� ������� ����� ������ ���������� �������
if df_full is not None:
    print(f"����� ������� ������: {len(df_full)}")


# ������� ��������� (���� ��� ���-�� ����������)
hubs_full = df_full.drop_duplicates(subset='URL')

# ��������� �������� ��������� �� �������� �� ��� ������
hubs_full.to_parquet('hubs_to_articles_urls.parquet', index=False)

# �������� �� 5 ������ (��� ����� ����� ��������� ������ ������� ������)
urls = hubs_full['URL']
hubs_parts = utils.split_list(urls, 5)

# ��������� ����� ������� � ����������
global_counter = [0]
global_lock = asyncio.Lock()


# ��������� ������� ��������� ������
for i, urls_chunk in enumerate(hubs_parts, 1):
    print(f"���������� ��������� ����� {i} �� {len(hubs_parts)}")
    asyncio.run(utils.process_part(urls_chunk, i, global_counter, global_lock))


# ������ �������� ��������� �� 5 ��������� ������
fin_df = pd.DataFrame()
for part in range(1, len(hubs_parts)+1):
    df_part = pd.read_parquet(f'articles_part_{part}.parquet')
    fin_df = pd.concat([fin_df, df_part], ignore_index=True)


fin_df.info()

# ������� ���������� ��� ��������� ������ (��� ��� Semaphore = 50) URL
# ��������� �� � ��������� � �������� ���������
url_dif = list(set(urls) - set(fin_df['URL']))
missed_articles = asyncio.run(utils.parse_article(url_dif, global_counter, global_lock))
missed_articles.to_parquet('missed_articles.parquet', index=False)
fin_df = pd.concat([fin_df, missed_articles], ignore_index=True)

# ���������� ������
fin_df.to_parquet('habr_articles_parsed_final.parquet', index=False)
