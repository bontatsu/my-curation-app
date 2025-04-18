# curation_logic.py (DB対応版)
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
# from sumy.summarizers.text_rank import TextRankSummarizer # 必要ならコメント解除
# from sumy.summarizers.lsa import LsaSummarizer # 必要ならコメント解除
from sumy.nlp.stemmers import Stemmer
import string
from bs4 import BeautifulSoup
# ★★★ DBからフィード情報を取得するために追加 ★★★
# (curation_app.py と同じ接続名を使う想定)
from sqlalchemy import text, exc as sqlalchemy_exc
DB_CONNECTION_NAME = "supabase_db"
# ★★★ ここまで追加 ★★★


print("curation_logic.py を読み込み中...")

# --- 設定値 ---
# FEED_CONFIG_FILE = 'feeds.json' # 不要になったためコメントアウト or 削除
# FEED_INFO = [] # 不要
# FEED_URLS = [] # 不要
# FEED_MAP = {} # 不要

PROCESSED_URLS_FILE = 'processed_urls.txt'
SIMILARITY_THRESHOLD = 0.9
SUMMARY_MAX_SENTENCES = 3
SUMMARY_MAX_LENGTH = 100
NUM_KEYWORDS = 5
LANGUAGE = "japanese"
STOP_WORDS_JA = {'する', 'いる', 'なる', 'ある', 'できる', 'ない', '良い', 'いう', '思う', 'もの', 'こと', 'ため', 'さん', 'よう', 'みたい', 'こちら'}
STOP_WORDS_EN = { 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'to', 'of', 'and', 'in', 'on', 'at', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'll', 're', 've', 'it', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that', 'these', 'those', 'am', 'has', 'have', 'had', 'having', 'do', 'does', 'did', 'doing', 'github', 'google' }

# --- Geminiクライアント初期化 (logic側でも行う) ---
GEMINI_LOGIC_INITIALIZED = False
try:
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
     print("!!! (Logic) Warning: st.secrets が利用できません (Streamlit環境外？)。環境変数など他の方法でAPIキーを読み込む必要があります。 !!!")
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


# ★★★ fetch_new_articles 関数 (変更なし、引数 feed_urls を受け取る) ★★★
def fetch_new_articles(feed_urls: list, processed_urls: set):
    """RSSフィードから新しい記事情報を収集し、日時がない場合は取得日時で補完する"""
    print("  - 新しい記事の収集を開始します...")
    newly_added_articles = []
    current_processed_urls = processed_urls.copy()

    if not feed_urls:
        print("    - 収集対象のフィードURLがありません。")
        return newly_added_articles, current_processed_urls

    for feed_url in feed_urls:
        if not feed_url: continue # URLが空ならスキップ
        print(f"    - フィードをチェック中: {feed_url[:80]}...")
        try:
            feed = feedparser.parse(feed_url)
            bozo_exception = feed.get("bozo_exception")
            if bozo_exception: print(f"      警告: フィード形式問題の可能性 ({feed_url}) - {bozo_exception}")

            for entry in feed.entries:
                article_link = getattr(entry, 'link', None)
                if not article_link: continue

                if article_link not in current_processed_urls:
                    fetch_time_gmt_entry = time.gmtime()
                    fetch_datetime_str_entry = time.strftime('%Y-%m-%d %H:%M:%S GMT', fetch_time_gmt_entry)
                    print(f"      - New article found: {article_link} (Fetched at: {fetch_datetime_str_entry})")

                    published_time_struct = getattr(entry, 'published_parsed', None)
                    published_datetime_str = None
                    if published_time_struct:
                        try: published_datetime_str = time.strftime('%Y-%m-%d %H:%M:%S GMT', published_time_struct)
                        except Exception as e_time: print(f"        - 警告: 日時変換エラー: {e_time}")

                    final_published_datetime_str = published_datetime_str if published_datetime_str else fetch_datetime_str_entry
                    if published_datetime_str is None: print("        - Published date is None. Using fetch date instead.")

                    raw_title = getattr(entry, 'title', 'タイトルなし'); raw_summary = getattr(entry, 'summary', '')
                    clean_title = remove_html_tags_bs4(raw_title); clean_summary = remove_html_tags_bs4(raw_summary)

                    article_info = {
                        'title': clean_title, 'link': article_link,
                        'published_datetime_str': final_published_datetime_str,
                        'summary': clean_summary, 'source_feed': feed_url, 'id': getattr(entry, 'id', article_link)
                    }
                    newly_added_articles.append(article_info)
                    current_processed_urls.add(article_link)
        except Exception as e: print(f"      フィード取得中に予期せぬエラー ({feed_url}): {e}")
    print(f"  - 収集完了。今回新たに追加された記事: {len(newly_added_articles)} 件")
    return newly_added_articles, current_processed_urls


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
        # elif algorithm == 'textrank': summarizer = TextRankSummarizer(stemmer) # 必要ならコメント解除
        # elif algorithm == 'lsa': summarizer = LsaSummarizer(stemmer) # 必要ならコメント解除
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

# --- Embedding API 呼び出し関数 (変更なし) ---
def get_embedding(text_or_texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    与えられたテキストまたはテキストリストからGemini Embeddingを取得する。
    失敗した場合は None を返す。
    """
    global GEMINI_LOGIC_INITIALIZED, EMBEDDING_MODEL_NAME
    if not GEMINI_LOGIC_INITIALIZED: print("    - Error: Gemini クライアント未初期化"); return None
    if not text_or_texts: return None
    try:
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=text_or_texts, task_type=task_type)
        return result['embedding']
    except Exception as e: print(f"    - Error: Gemini Embedding API呼び出しエラー: {e}"); return None

# --- メイン処理パイプライン関数 (引数追加) ---
# ★★★ 引数 feed_list を追加 ★★★
def run_curation_pipeline(feed_list: list):
    # global FEED_URLS # 不要になったため削除
    print("\n=== 情報キュレーションパイプライン開始 ===")
    start_time = time.time()

    # ★★★ 引数 feed_list から URL リストを生成 ★★★
    feed_urls_to_process = []
    if isinstance(feed_list, list):
        feed_urls_to_process = [item.get('url') for item in feed_list if item.get('url')]
    if not feed_urls_to_process:
        print("  - 処理対象のフィードURLがありません。パイプラインを終了します。")
        return []
    print(f"  - 処理対象フィードURL数: {len(feed_urls_to_process)}")
    # ★★★ ここまで変更 ★★★

    processed_urls = load_processed_urls(PROCESSED_URLS_FILE)
    # ★★★ 生成した feed_urls_to_process を fetch_new_articles に渡す ★★★
    newly_added_articles, updated_processed_urls = fetch_new_articles(feed_urls_to_process, processed_urls)

    # --- 以降の処理は変更なし ---
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
    # ★★★ 直接実行テストではDBからフィードを取得する必要がある ★★★
    test_feeds = []
    try:
        # このテスト実行が Streamlit 環境外で行われる場合、st.connection は使えない可能性がある
        # 代替として、直接 DB に接続するか、テスト用の固定リストを使うなどの工夫が必要
        # ここでは、st.connection が使える前提で試みる (Streamlit コンテキスト内で実行される場合)
        print("  - テスト用にDBからフィードリストを取得します...")
        conn = st.connection(DB_CONNECTION_NAME, type="sql")
        df_feeds = conn.query("SELECT url, name FROM feeds ORDER BY name")
        test_feeds = df_feeds.to_dict('records')
        print(f"    - {len(test_feeds)} 件のフィードを取得しました。")
    except Exception as e_test_db:
        print(f"  - Error: テスト実行時のDBからのフィード取得に失敗: {e_test_db}")
        print("  - テスト実行をスキップします。")

    if test_feeds:
        results = run_curation_pipeline(feed_list=test_feeds) # ★引数を渡す
        print(f"\n--- 実行結果 ({len(results)} 件のユニーク記事) ---")
    else:
        print("  - 処理対象のフィードがないため、テスト実行をスキップしました。")

    # --- Embedding関数のテスト ---
    print("\n--- Embedding Function Test ---")
    test_text_1 = "これは最初のテスト文章です。"
    test_text_2 = "これは二番目の文章、少し違います。"
    test_texts = [test_text_1, test_text_2]
    print(f"Testing single text: '{test_text_1}'")
    embedding1 = get_embedding(test_text_1)
    if embedding1: print(f"  -> Got embedding vector of dimension: {len(embedding1)}")
    else: print("  -> Failed to get embedding.")
    print(f"\nTesting text list: {test_texts}")
    embeddings_list = get_embedding(test_texts)
    if embeddings_list and isinstance(embeddings_list, list) and len(embeddings_list) == len(test_texts): print(f"  -> Got {len(embeddings_list)} embedding vectors. Dim: {len(embeddings_list[0])}")
    else: print("  -> Failed to get embeddings for the list.")
    print("--- Embedding Function Test End ---")

    print("\n--- 直接実行テスト完了 ---")

