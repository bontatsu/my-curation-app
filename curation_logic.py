# curation_logic.py (公開日時補完版)
# curation_logic.py の冒頭に追加
import streamlit as st # Secrets 読み込みのため
import google.generativeai as genai
# 他の既存の import 文 ...
import feedparser
import time
import os
import re
import json
import numpy as np
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
import string
from bs4 import BeautifulSoup


print("curation_logic.py を読み込み中...")

# --- 設定値 ---
FEED_CONFIG_FILE = 'feeds.json'
FEED_INFO = []
FEED_URLS = []
FEED_MAP = {}
if os.path.exists(FEED_CONFIG_FILE):
    try:
        with open(FEED_CONFIG_FILE, 'r', encoding='utf-8') as f: FEED_INFO = json.load(f)
        if isinstance(FEED_INFO, list):
            FEED_URLS = [item.get('url') for item in FEED_INFO if item.get('url')]
            FEED_MAP = {item.get('url'): item.get('name', item.get('url')) for item in FEED_INFO if item.get('url')}
            print(f"  - {FEED_CONFIG_FILE} から {len(FEED_URLS)} 件のフィード情報を読み込みました。")
        else: print(f"  - 警告: {FEED_CONFIG_FILE} の形式が不正です。"); FEED_INFO = []
    except Exception as e: print(f"  - Error: {FEED_CONFIG_FILE} の読み込みエラー: {e}"); FEED_INFO = []
else: print(f"  - 警告: フィード設定ファイル '{FEED_CONFIG_FILE}' が見つかりません。")
if not FEED_URLS: print("  - 警告: 処理対象のフィードURLがありません。")

PROCESSED_URLS_FILE = 'processed_urls.txt'
SIMILARITY_THRESHOLD = 0.85
SUMMARY_MAX_SENTENCES = 3
SUMMARY_MAX_LENGTH = 100
NUM_KEYWORDS = 5
LANGUAGE = "japanese"
STOP_WORDS_JA = {'する', 'いる', 'なる', 'ある', 'できる', 'ない', '良い', 'いう', '思う', 'もの', 'こと', 'ため', 'さん', 'よう', 'みたい', 'こちら'}
STOP_WORDS_EN = { 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'll', 're', 've', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that', 'these', 'those', 'am', 'has', 'have', 'had', 'having', 'do', 'does', 'did', 'doing', 'github', 'google' }

# curation_logic.py の設定値定義の後などに追加

# --- Geminiクライアント初期化 (logic側でも行う) ---
GEMINI_LOGIC_INITIALIZED = False
try:
    # StreamlitのSecrets機能を使ってAPIキーを読み込む
    # secrets.toml が存在し、キーが設定されている必要がある
    if "GEMINI_API_KEY" in st.secrets:
         api_key = st.secrets["GEMINI_API_KEY"]
         genai.configure(api_key=api_key)
         print("  - (Logic) Gemini クライアントの初期化成功。")
         GEMINI_LOGIC_INITIALIZED = True
    else:
         print("!!! (Logic) Secrets Warning: secrets.toml に GEMINI_API_KEY が見つかりません。 !!!")
except FileNotFoundError:
     print("!!! (Logic) Secrets Warning: secrets.toml ファイルが見つかりません。 !!!")
except AttributeError:
     # st.secrets が Streamlitサーバープロセス外（例: 直接実行時）で使えない場合
     print("!!! (Logic) Warning: st.secrets が利用できません (Streamlit環境外？)。環境変数など他の方法でAPIキーを読み込む必要があります。 !!!")
     # ここで環境変数から読み込む代替処理などを追加しても良い
     # api_key = os.environ.get("GEMINI_API_KEY")
     # if api_key: genai.configure(api_key=api_key); GEMINI_LOGIC_INITIALIZED = True ...
except Exception as e:
    print(f"!!! (Logic) Geminiクライアント初期化エラー: {e} !!!")

# 使用するEmbeddingモデル名
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # 無料のモデルを指定

# --- 関数定義 ---
try: janome_tokenizer = Tokenizer(); print("  - Janome Tokenizer の初期化成功。")
except Exception as e: print(f"  - Janome Tokenizer 初期化エラー: {e}"); janome_tokenizer = None

def remove_html_tags_bs4(text): # ... (変更なし) ...
    if not isinstance(text, str) or not text.strip(): return ""
    try: soup = BeautifulSoup(text, 'lxml'); clean_text = soup.get_text(separator=' ', strip=True); clean_text = re.sub(r'\s+', ' ', clean_text); return clean_text
    except Exception as e: print(f"    - Warning: BeautifulSoupでのHTML除去エラー: {e}"); clean = re.compile('<.*?>'); return re.sub(clean, '', text).strip()

def load_processed_urls(filepath): # ... (変更なし) ...
    processed_urls = set();
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f: processed_urls = {line.strip() for line in f if line.strip()}
            print(f"  - {filepath} から {len(processed_urls)} 件の処理済みURLを読み込みました。")
        except Exception as e: print(f"  - 処理済みURLファイルの読み込み中にエラー: {e}")
    else: print(f"  - {filepath} が見つかりません。新規作成されます。")
    return processed_urls

def save_processed_urls(filepath, urls_set): # ... (変更なし) ...
    print(f"  - {filepath} に {len(urls_set)} 件の処理済みURLを保存します...")
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for url in sorted(list(urls_set)): f.write(url + '\n')
        print("  - 保存完了。")
    except Exception as e: print(f"  - 処理済みURLファイルの保存中にエラー: {e}")


# ★★★ fetch_new_articles 関数を修正 ★★★
def fetch_new_articles(feed_urls, processed_urls):
    """RSSフィードから新しい記事情報を収集し、日時がない場合は取得日時で補完する"""
    print("  - 新しい記事の収集を開始します...")
    newly_added_articles = []
    current_processed_urls = processed_urls.copy()

    for feed_url in feed_urls:
        print(f"    - フィードをチェック中: {feed_url[:80]}...")
        try:
            feed = feedparser.parse(feed_url)
            bozo_exception = feed.get("bozo_exception")
            if bozo_exception: print(f"      警告: フィード形式問題の可能性 ({feed_url}) - {bozo_exception}")

            for entry in feed.entries:
                article_link = getattr(entry, 'link', None)
                if not article_link: continue

                if article_link not in current_processed_urls:
                    # 記事発見時の取得日時を記録
                    fetch_time_gmt_entry = time.gmtime()
                    fetch_datetime_str_entry = time.strftime('%Y-%m-%d %H:%M:%S GMT', fetch_time_gmt_entry)
                    print(f"      - New article found: {article_link} (Fetched at: {fetch_datetime_str_entry})")

                    # 元の公開日時を取得・変換
                    published_time_struct = getattr(entry, 'published_parsed', None)
                    published_datetime_str = None
                    if published_time_struct:
                        try: published_datetime_str = time.strftime('%Y-%m-%d %H:%M:%S GMT', published_time_struct)
                        except Exception as e_time: print(f"        - 警告: 日時変換エラー: {e_time}")

                    # 公開日時がNoneなら取得日時で補完
                    final_published_datetime_str = published_datetime_str
                    if final_published_datetime_str is None:
                        print("        - Published date is None. Using fetch date instead.")
                        final_published_datetime_str = fetch_datetime_str_entry

                    # title と summary の処理
                    raw_title = getattr(entry, 'title', 'タイトルなし'); raw_summary = getattr(entry, 'summary', '')
                    clean_title = remove_html_tags_bs4(raw_title); clean_summary = remove_html_tags_bs4(raw_summary)

                    # article_info 辞書作成 (補完後の日時を使用)
                    article_info = {
                        'title': clean_title, 'link': article_link,
                        'published_datetime_str': final_published_datetime_str, # 補完後の日時
                        'summary': clean_summary, 'source_feed': feed_url, 'id': getattr(entry, 'id', article_link)
                    }
                    newly_added_articles.append(article_info)
                    current_processed_urls.add(article_link)
        except Exception as e: print(f"      フィード取得中に予期せぬエラー ({feed_url}): {e}")
    print(f"  - 収集完了。今回新たに追加された記事: {len(newly_added_articles)} 件")
    return newly_added_articles, current_processed_urls
# ★★★ fetch_new_articles 関数ここまで ★★★


def tokenize_and_filter(text): # ... (変更なし) ...
    tokens = []; global janome_tokenizer, STOP_WORDS_JA, STOP_WORDS_EN
    if not janome_tokenizer: return tokens
    if not isinstance(text, str): text = ""
    text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text); text = re.sub(r'[0-9０-９]+', ' ', text); text = text.strip(); text = re.sub(r'\s+', ' ', text)
    if not text: return tokens
    try:
        for token in janome_tokenizer.tokenize(text):
            part_of_speech = token.part_of_speech.split(',')[0]; base_form = token.base_form; surface = token.surface; should_remove = False
            if part_of_speech not in ['名詞', '動詞', '形容詞']: should_remove = True
            else:
                if base_form in STOP_WORDS_JA: should_remove = True
                elif base_form.lower() in STOP_WORDS_EN: should_remove = True
                elif part_of_speech == '名詞' and len(base_form) == 1 and base_form.upper() not in ('A', 'B', 'C', 'X', 'I'): should_remove = True
                elif base_form.isdigit(): should_remove = True
                elif all(c in string.punctuation or c.isspace() for c in base_form) and base_form: should_remove = True
            if not should_remove: tokens.append(base_form)
    except Exception as e: print(f"    - 形態素解析中にエラー: {e} - Text: {text[:50]}...")
    return tokens

def get_vectorizer(): # ... (変更なし) ...
    print("    - TfidfVectorizer 設定: min_df=1, max_df=1.0"); return TfidfVectorizer(tokenizer=tokenize_and_filter, token_pattern=None, min_df=1, max_df=1.0)

def calculate_similarity(texts): # ... (変更なし) ...
    print(f"  - {len(texts)} 件の記事テキストで類似度を計算します...")
    vectorizer = get_vectorizer(); tfidf_matrix = None; similarity_matrix = None
    if not texts: print("    - テキストがないため計算をスキップします。"); return vectorizer, tfidf_matrix, similarity_matrix
    try:
        tfidf_matrix = vectorizer.fit_transform(texts); print(f"    - TF-IDF行列生成完了 (形状: {tfidf_matrix.shape})")
        if tfidf_matrix.shape[0] > 1: similarity_matrix = cosine_similarity(tfidf_matrix); print(f"    - コサイン類似度行列生成完了 (形状: {similarity_matrix.shape})")
        else: print("    - 記事が1件以下のため、コサイン類似度計算はスキップします。")
    except Exception as e: print(f"    - 類似度計算中にエラーが発生しました: {e}"); tfidf_matrix = None; similarity_matrix = None
    return vectorizer, tfidf_matrix, similarity_matrix

def find_duplicates(similarity_matrix, num_articles_processed, threshold=0.85): # ... (変更なし) ...
    unique_indices = []; duplicate_indices = set()
    if similarity_matrix is None or num_articles_processed < 2: print(f"    - 類似度計算スキップまたは記事数不足のため、全 {num_articles_processed} 件をユニーク扱いします。"); unique_indices = list(range(num_articles_processed)); return unique_indices, duplicate_indices
    print(f"    - 類似度閾値 {threshold} で重複記事を判定します..."); num_articles = num_articles_processed; processed_flags = [False] * num_articles
    for i in range(num_articles):
        if processed_flags[i]: continue
        unique_indices.append(i); processed_flags[i] = True
        for j in range(i + 1, num_articles):
            if not processed_flags[j] and i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1] and similarity_matrix[i, j] >= threshold: processed_flags[j] = True; duplicate_indices.add(j)
    print(f"      重複判定完了。ユニーク: {len(unique_indices)} 件, 重複: {len(duplicate_indices)} 件"); return unique_indices, duplicate_indices

def summarize_lead_sentences(text, max_sentences=3, max_length=100): # ... (変更なし) ...
    if not isinstance(text, str) or not text.strip(): return ""
    summary = ""; sentences_count = 0
    try:
        sentences = re.split(r'([。!?])\s*', text); processed_sentences = []
        temp_sentence = "";
        for part in sentences:
            if not part: continue
            temp_sentence += part
            if part in ['。', '!', '?']: processed_sentences.append(temp_sentence.strip()); temp_sentence = ""
        if temp_sentence: processed_sentences.append(temp_sentence.strip())
        processed_sentences = [s for s in processed_sentences if s]; summary_sentences = []; current_length = 0
        for sentence in processed_sentences:
            if current_length + len(sentence) > max_length and len(summary_sentences) > 0: break
            if len(summary_sentences) < max_sentences: summary_sentences.append(sentence); current_length += len(sentence)
            else: break
        summary = " ".join(summary_sentences); sentences_count = len(summary_sentences)
    except Exception as e: print(f"    - リード文要約中のエラー: {e}"); summary = text[:max_length] + "..." if len(text) > max_length else text
    return summary

def summarize_text_with_sumy(text, sentences_count=3, algorithm='lexrank'): # ... (変更なし) ...
    global janome_tokenizer;
    if not isinstance(text, str) or not text.strip() or not janome_tokenizer: return ""
    summary = ""; sentences_count_actual = 0
    try:
        parser = PlaintextParser.from_string(text, SumyTokenizer(LANGUAGE)); stemmer = Stemmer(LANGUAGE)
        if algorithm == 'lexrank': summarizer = LexRankSummarizer(stemmer)
        elif algorithm == 'textrank': summarizer = TextRankSummarizer(stemmer)
        elif algorithm == 'lsa': summarizer = LsaSummarizer(stemmer)
        else: summarizer = LexRankSummarizer(stemmer)
        summary_sentences_tuples = summarizer(parser.document, sentences_count)
        summary_sentences = [str(sentence) for sentence in summary_sentences_tuples]
        summary = " ".join(summary_sentences); sentences_count_actual = len(summary_sentences)
    except Exception as e: print(f"    - sumyでの要約中にエラーが発生しました: {e}"); summary = summarize_lead_sentences(text, sentences_count, SUMMARY_MAX_LENGTH); print("      -> フォールバックとしてリード文要約を使用")
    return summary

def extract_keywords_for_articles(vectorizer, tfidf_matrix, target_indices, num_keywords=5): # ... (変更なし) ...
    keywords_dict = {}
    if vectorizer is None or tfidf_matrix is None or not target_indices: print("    - キーワード抽出に必要な情報がないためスキップします。"); return keywords_dict
    print(f"  - {len(target_indices)} 件の記事について上位 {num_keywords} 個のキーワードを抽出します...")
    feature_names = None; extracted_count = 0
    try: feature_names = vectorizer.get_feature_names_out()
    except Exception as e: print(f"    - 語彙リスト取得エラー: {e}"); return keywords_dict
    if feature_names is None or len(feature_names) == 0: print(f"    - 語彙リストが空のためキーワード抽出をスキップします。"); return keywords_dict
    for matrix_idx in target_indices:
        keywords = []
        try:
            if matrix_idx < tfidf_matrix.shape[0]:
                row_vector = tfidf_matrix.getrow(matrix_idx)
                if row_vector.nnz > 0:
                    col_indices = row_vector.indices; scores = row_vector.data; sorted_col_indices_by_score = col_indices[np.argsort(scores)[::-1]]
                    top_n_indices = sorted_col_indices_by_score[:num_keywords]
                    keywords = [feature_names[idx] for idx in top_n_indices if idx < len(feature_names)]
        except IndexError: print(f"    - キーワード抽出中にインデックスエラー (matrix_idx {matrix_idx})")
        except Exception as e: print(f"    - キーワード抽出中に予期せぬエラー (matrix_idx {matrix_idx}): {e}")
        keywords_dict[matrix_idx] = keywords
        if keywords: extracted_count += 1
    print(f"    - キーワード抽出完了 ({extracted_count} 件の記事でキーワードを抽出)。")
    return keywords_dict

# curation_logic.py に追加

# --- 3-9. Embedding API 呼び出し関数 ---
def get_embedding(text_or_texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    与えられたテキストまたはテキストリストからGemini Embeddingを取得する。
    失敗した場合は None を返す。

    Args:
        text_or_texts (str or list[str]): ベクトル化したいテキスト、またはそのリスト。
        task_type (str): Embeddingのタスクタイプ。検索対象文書なら "RETRIEVAL_DOCUMENT",
                         検索クエリなら "RETRIEVAL_QUERY", 分類なら "CLASSIFICATION" など。

    Returns:
        list[float] or list[list[float]] or None: Embeddingベクトル、またはそのリスト。エラー時はNone。
    """
    global GEMINI_LOGIC_INITIALIZED, EMBEDDING_MODEL_NAME # 初期化状態とモデル名を参照

    if not GEMINI_LOGIC_INITIALIZED:
        print("    - Error: Gemini クライアントが初期化されていないため、Embeddingを取得できません。")
        return None
    if not text_or_texts: # 入力が空の場合
         return None

    try:
        # content にテキストまたはリストをそのまま渡せる
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text_or_texts,
            task_type=task_type
        )
        # result['embedding'] にベクトル(リスト)またはベクトルのリストが入っている
        return result['embedding']
    except Exception as e:
        print(f"    - Error: Gemini Embedding API呼び出し中にエラーが発生しました: {e}")
        # エラーの詳細を知りたい場合は traceback を使う
        # import traceback
        # print(traceback.format_exc())
        return None

# --- メイン処理パイプライン関数 ---
def run_curation_pipeline(): # ★★★ この関数内で fetch_new_articles が呼ばれる ★★★
    global FEED_URLS # logicファイル内で生成された FEED_URLS を使う
    print("\n=== 情報キュレーションパイプライン開始 ===")
    start_time = time.time()
    processed_urls = load_processed_urls(PROCESSED_URLS_FILE)
    newly_added_articles, updated_processed_urls = fetch_new_articles(FEED_URLS, processed_urls) # ★ 修正済みの関数呼び出し ★
    if not newly_added_articles:
        print("新しい記事はありませんでした。"); save_processed_urls(PROCESSED_URLS_FILE, updated_processed_urls)
        print(f"パイプライン完了 ({(time.time() - start_time):.2f}秒)"); return []
    texts_for_similarity = []; original_indices = []
    for i, article in enumerate(newly_added_articles):
        text = f"{article.get('title', '')} {article.get('summary', '')}".strip()
        if text: texts_for_similarity.append(text); original_indices.append(i)
        else: print(f"  警告: 元記事インデックス {i} はテキストが空のため類似度計算から除外。")
    if not texts_for_similarity:
        print("類似度計算対象のテキストがありません。"); save_processed_urls(PROCESSED_URLS_FILE, updated_processed_urls)
        print(f"パイプライン完了 ({(time.time() - start_time):.2f}秒)"); return []
    vectorizer, tfidf_matrix, similarity_matrix = calculate_similarity(texts_for_similarity)
    unique_indices_in_matrix, duplicate_indices_in_matrix = find_duplicates(similarity_matrix, len(texts_for_similarity), SIMILARITY_THRESHOLD)
    processed_unique_articles = []; matrix_idx_to_original_idx = {matrix_idx: original_idx for matrix_idx, original_idx in enumerate(original_indices)}
    for matrix_idx in unique_indices_in_matrix:
        original_idx = matrix_idx_to_original_idx.get(matrix_idx)
        if original_idx is not None and original_idx < len(newly_added_articles):
            article_copy = newly_added_articles[original_idx].copy(); article_copy['_matrix_idx'] = matrix_idx; processed_unique_articles.append(article_copy)
        else: print(f"  警告: matrix_idx {matrix_idx} に対応する元記事が見つかりません。")
    print("  - sumy を使った要約を生成します...")
    generated_summary_count = 0
    for article in processed_unique_articles:
        text_to_summarize = f"{article.get('title', '')} {article.get('summary', '')}".strip()
        article['summary_generated'] = summarize_text_with_sumy(text_to_summarize, sentences_count=SUMMARY_MAX_SENTENCES, algorithm='lexrank')
        if article['summary_generated']: generated_summary_count += 1
        if 'lead_summary' in article: del article['lead_summary']
    print(f"    - 要約生成完了 ({generated_summary_count} 件)。")
    keywords_for_unique_articles = extract_keywords_for_articles(vectorizer, tfidf_matrix, unique_indices_in_matrix, NUM_KEYWORDS)
    merged_keyword_count = 0
    for article in processed_unique_articles:
        matrix_idx = article.get('_matrix_idx')
        if matrix_idx is not None:
            article['keywords_tfidf'] = keywords_for_unique_articles.get(matrix_idx, [])
            if article['keywords_tfidf']: merged_keyword_count += 1
            if '_matrix_idx' in article: del article['_matrix_idx']
        else: article['keywords_tfidf'] = []
    print(f"  - キーワードを {merged_keyword_count} 件の記事に追加しました。")
    save_processed_urls(PROCESSED_URLS_FILE, updated_processed_urls)
    end_time = time.time()
    print(f"=== 情報キュレーションパイプライン完了: {len(processed_unique_articles)} 件の新規ユニーク記事を処理 ({(end_time - start_time):.2f}秒) ===")
    return processed_unique_articles

# --- メイン処理の実行 (直接実行用) ---
if __name__ == '__main__':
    print("\n--- curation_logic.py を直接実行テスト ---")
    if FEED_MAP: results = run_curation_pipeline(); print(f"\n--- 実行結果 ({len(results)} 件のユニーク記事) ---")
    else: print("FEED_MAPが空のため、テスト実行をスキップしました。feeds.jsonを確認してください。")
    print("\n--- 直接実行テスト完了 ---")
# curation_logic.py の if __name__ == '__main__': ブロック内に追加

    # --- Embedding関数のテスト ---
    print("\n--- Embedding Function Test ---")
    test_text_1 = "これは最初のテスト文章です。"
    test_text_2 = "これは二番目の文章、少し違います。"
    test_texts = [test_text_1, test_text_2]

    # 単一テキストのテスト
    print(f"Testing single text: '{test_text_1}'")
    embedding1 = get_embedding(test_text_1)
    if embedding1:
        print(f"  -> Got embedding vector of dimension: {len(embedding1)}")
        # print(f"  -> Vector (first 5 dims): {embedding1[:5]}") # ベクトルの中身（一部）
    else:
        print("  -> Failed to get embedding.")

    # テキストリストのテスト
    print(f"\nTesting text list: {test_texts}")
    embeddings_list = get_embedding(test_texts)
    if embeddings_list and isinstance(embeddings_list, list) and len(embeddings_list) == len(test_texts):
        print(f"  -> Got {len(embeddings_list)} embedding vectors.")
        print(f"  -> Dimension of first vector: {len(embeddings_list[0])}")
        # print(f"  -> First vector (first 5 dims): {embeddings_list[0][:5]}")
    else:
        print("  -> Failed to get embeddings for the list.")
    print("--- Embedding Function Test End ---")

else: # FEED_MAP が空の場合
     print("FEED_MAPが空のため、テスト実行をスキップしました。feeds.jsonを確認してください。")
print("\n--- 直接実行テスト完了 ---")