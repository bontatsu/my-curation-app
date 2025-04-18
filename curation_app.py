# curation_app.py (Signal Spotter - キャッシュクリア修正版)

import streamlit as st
import pandas as pd
import json
import os
import time
import traceback
import google.generativeai as genai
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# SQLAlchemy のエラークラスと text をインポート
from sqlalchemy import text, exc as sqlalchemy_exc

# --- 定数定義 ---
DB_CONNECTION_NAME = "supabase_db" # secrets.toml の接続名
RECOMMENDATION_BOOST_WEIGHT = 0.5
RECOMMENDATION_COUNT = 5
RECOMMENDATION_THRESHOLD_SCORE = 0.4
INITIAL_ARTICLE_LOAD_LIMIT = 200
# Session State のキーを定数化
SESSION_KEY_ARTICLES = "articles_data"
SESSION_KEY_LAST_INTERACTED = "last_interacted_link"
SESSION_KEY_INTEREST_KEYWORDS = "user_interest_keywords"
SESSION_KEY_SEARCH_BOX = "sidebar_search_box"
SESSION_KEY_SOURCE_FILTER = "sidebar_source_filter_names"
SESSION_KEY_FORCE_REFRESH = "force_refresh_articles"
# SESSION_KEY_FEED_VERSION = "feed_list_version" # 不要になったため削除
# SESSION_KEY_KEYWORD_VERSION = "keyword_list_version" # 不要になったため削除

# --- curation_logic からインポート ---
run_curation_pipeline = None
get_embedding = None
IMPORT_SUCCESS = False
IMPORT_ERROR_MESSAGE = ""
try:
    # ★★★ curation_logic_db (修正版) をインポートするようにしてください ★★★
    from curation_logic import run_curation_pipeline, get_embedding
    IMPORT_SUCCESS = True
    print("curation_logic.py のインポート成功。")
except ImportError as e:
    IMPORT_ERROR_MESSAGE = f"curation_logic.py が見つからないか、インポート中にエラーが発生しました: {e}"
    print(f"!!! ImportError: {IMPORT_ERROR_MESSAGE} !!!")
except Exception as e_general:
    IMPORT_ERROR_MESSAGE = f"curation_logic.py の読み込み中に予期せぬエラーが発生しました: {e_general}\n{traceback.format_exc()}"
    print(f"!!! Exception during import: {IMPORT_ERROR_MESSAGE} !!!")

# --- Streamlit ページ設定 ---
st.set_page_config(page_title="Signal Spotter", layout="wide", page_icon="📰")

# --- Session State 初期化 (バージョンカウンター削除) ---
# if SESSION_KEY_FEED_VERSION not in st.session_state:
#     st.session_state[SESSION_KEY_FEED_VERSION] = 0
# if SESSION_KEY_KEYWORD_VERSION not in st.session_state:
#     st.session_state[SESSION_KEY_KEYWORD_VERSION] = 0

# --- Geminiクライアント初期化 ---
GEMINI_INITIALIZED = False
if IMPORT_SUCCESS:
    try:
        if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            print("Gemini クライアント初期化成功。")
            GEMINI_INITIALIZED = True
        elif not hasattr(st, 'secrets'):
             print("!!! Warning: st.secrets が利用できません。")
        else:
            print("!!! Secrets Warning: GEMINI_API_KEY が secrets.toml に設定されていません。")
    except FileNotFoundError:
        print("!!! Secrets Warning: secrets.toml ファイルが見つかりません。")
    except AttributeError:
        print("!!! Warning: st.secrets 属性が利用できません。")
    except Exception as e:
        print(f"!!! Gemini 初期化中に予期せぬエラー: {e} !!!")
        print(traceback.format_exc())
        st.error(f"Geminiクライアントの初期化に失敗しました: {e}")

# --- DB初期化/接続確認関数 ---
def init_db(conn_name=DB_CONNECTION_NAME):
    """データベース接続と 'articles', 'feeds', 'interest_keywords' テーブルの存在を確認する"""
    print(f"データベース接続を確認します ({conn_name})...")
    try:
        conn = st.connection(conn_name, type="sql")
        # articles テーブルの確認
        conn.query("SELECT 1 FROM articles LIMIT 1", ttl=0)
        print("  - articles テーブル OK.")
        # feeds テーブルの確認
        conn.query("SELECT 1 FROM feeds LIMIT 1", ttl=0)
        print("  - feeds テーブル OK.")
        # interest_keywords テーブルの確認
        conn.query("SELECT 1 FROM interest_keywords LIMIT 1", ttl=0)
        print("  - interest_keywords テーブル OK.")
        print("  - データベース接続および必要テーブルの確認OK。")
        return True
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy データベース接続またはテーブル確認エラー: {e} !!!")
        st.error(f"データベース接続またはテーブル確認エラー: {e}。Supabase側でテーブル(articles, feeds, interest_keywords)が作成されているか確認してください。")
        return False
    except Exception as e:
        print(f"!!! 予期せぬデータベースエラー: {e} !!!")
        st.error(f"データベース接続中に予期せぬエラーが発生しました: {e}")
        return False

# --- ★★★ 設定情報DBアクセス関数 (キャッシュクリア方法変更) ★★★ ---

@st.cache_data # DBからの読み込み結果をキャッシュ
def load_feeds_from_db(conn_name=DB_CONNECTION_NAME): # ★ダミー引数 _version を削除
    """データベースの feeds テーブルからフィード情報を読み込む"""
    print(f"キャッシュ確認 or DB ({conn_name}) からフィード情報読み込み...")
    feeds = []
    try:
        conn = st.connection(conn_name, type="sql")
        df = conn.query("SELECT url, name FROM feeds ORDER BY name")
        feeds = df.to_dict('records')
        print(f"  - DBから {len(feeds)} 件のフィード情報を読み込み完了。")
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy DBエラー（フィード読み込み）: {e} !!!")
        st.error(f"データベースからのフィード情報読み込み中にエラー: {e}")
    except Exception as e:
        print(f"!!! 予期せぬエラー（フィード読み込み）: {e} !!!")
        st.error(f"フィード情報読み込み中に予期せぬエラー: {e}")
        traceback.print_exc()
    return feeds

@st.cache_data # DBからの読み込み結果をキャッシュ
def load_keywords_from_db(conn_name=DB_CONNECTION_NAME): # ★ダミー引数 _version を削除
    """データベースの interest_keywords テーブルからキーワードを読み込む"""
    print(f"キャッシュ確認 or DB ({conn_name}) から興味キーワード読み込み...")
    keywords = []
    try:
        conn = st.connection(conn_name, type="sql")
        df = conn.query("SELECT keyword FROM interest_keywords ORDER BY keyword")
        keywords = df['keyword'].tolist()
        print(f"  - DBから {len(keywords)} 個の興味キーワードを読み込み完了。")
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy DBエラー（キーワード読み込み）: {e} !!!")
        st.error(f"データベースからのキーワード読み込み中にエラー: {e}")
    except Exception as e:
        print(f"!!! 予期せぬエラー（キーワード読み込み）: {e} !!!")
        st.error(f"キーワード読み込み中に予期せぬエラー: {e}")
        traceback.print_exc()
    return keywords

def add_feed_to_db(name: str, url: str, conn_name=DB_CONNECTION_NAME):
    """データベースの feeds テーブルに新しいフィードを追加する"""
    print(f"DB ({conn_name}) にフィードを追加します: name='{name}', url='{url}'")
    success = False
    try:
        conn = st.connection(conn_name, type="sql")
        sql = text("INSERT INTO feeds (url, name) VALUES (:url, :name) ON CONFLICT (url) DO NOTHING")
        with conn.session as s:
            result = s.execute(sql, {"url": url, "name": name})
            s.commit()
            if result.rowcount > 0:
                print("  - フィードの追加成功。")
                success = True
            else:
                print("  - フィードは既に存在するため、追加されませんでした。")
                success = True # 既存でもエラーではない
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy DBエラー（フィード追加）: {e} !!!")
        st.error(f"フィードの追加中にデータベースエラーが発生しました: {e}")
    except Exception as e:
        print(f"!!! 予期せぬエラー（フィード追加）: {e} !!!")
        st.error(f"フィード追加中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()

    if success:
        # ★★★ 成功したらキャッシュを .clear() でクリア ★★★
        try:
            load_feeds_from_db.clear()
            print("  - load_feeds_from_db キャッシュをクリアしました。")
        except Exception as e_clear:
            print(f"!!! フィードキャッシュクリア中にエラー: {e_clear} !!!")
    return success

def delete_feed_from_db(url: str, conn_name=DB_CONNECTION_NAME):
    """データベースの feeds テーブルから指定されたURLのフィードを削除する"""
    print(f"DB ({conn_name}) からフィードを削除します: url='{url}'")
    success = False
    try:
        conn = st.connection(conn_name, type="sql")
        sql = text("DELETE FROM feeds WHERE url = :url")
        with conn.session as s:
            result = s.execute(sql, {"url": url})
            s.commit()
            if result.rowcount > 0:
                print("  - フィードの削除成功。")
                success = True
            else:
                print("  - Warn: 削除対象のフィードが見つかりませんでした。")
                success = False
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy DBエラー（フィード削除）: {e} !!!")
        st.error(f"フィードの削除中にデータベースエラーが発生しました: {e}")
    except Exception as e:
        print(f"!!! 予期せぬエラー（フィード削除）: {e} !!!")
        st.error(f"フィード削除中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()

    if success:
        # ★★★ 成功したらキャッシュを .clear() でクリア ★★★
        try:
            load_feeds_from_db.clear()
            print("  - load_feeds_from_db キャッシュをクリアしました。")
        except Exception as e_clear:
            print(f"!!! フィードキャッシュクリア中にエラー: {e_clear} !!!")
    return success

def save_keywords_to_db(keywords_list: list, conn_name=DB_CONNECTION_NAME):
    """データベースの interest_keywords テーブルを更新する (全削除 & 全挿入)"""
    print(f"DB ({conn_name}) の興味キーワードを更新します ({len(keywords_list)} 個)...")
    success = False
    try:
        conn = st.connection(conn_name, type="sql")
        with conn.session as s:
            # 1. 既存のキーワードをすべて削除
            s.execute(text("DELETE FROM interest_keywords"))
            print("  - 既存のキーワードを削除しました。")
            # 2. 新しいキーワードリストを挿入
            if keywords_list: # キーワードがある場合のみ挿入
                sql_insert = text("INSERT INTO interest_keywords (keyword) VALUES (:keyword)")
                params = [{"keyword": kw} for kw in keywords_list]
                s.execute(sql_insert, params)
                print(f"  - {len(keywords_list)} 個の新しいキーワードを挿入しました。")
            s.commit()
            success = True
            print("  - キーワードのDB更新完了。")
    except sqlalchemy_exc.SQLAlchemyError as e:
        print(f"!!! SQLAlchemy DBエラー（キーワード保存）: {e} !!!")
        st.error(f"キーワードの保存中にデータベースエラーが発生しました: {e}")
    except Exception as e:
        print(f"!!! 予期せぬエラー（キーワード保存）: {e} !!!")
        st.error(f"キーワード保存中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()

    if success:
        # ★★★ 成功したら関連キャッシュをクリア ★★★
        try:
            load_keywords_from_db.clear() # キーワード読み込みキャッシュをクリア
            print("  - load_keywords_from_db キャッシュをクリアしました。")
            get_interest_vector.clear() # 興味ベクトルキャッシュもクリア
            print("  - get_interest_vector キャッシュをクリアしました。")
        except Exception as e_clear:
            print(f"!!! キーワード関連キャッシュクリア中にエラー: {e_clear} !!!")
    return success

# --- マッピング関数 (変更なし) ---
def get_feed_map_from_list(feed_data_list):
    """フィード情報のリストから URL をキー、名前を値とする辞書を作成する"""
    feed_map = {}
    if isinstance(feed_data_list, list):
        for item in feed_data_list:
            url = item.get('url')
            name = item.get('name')
            if url:
                feed_map[url] = name if name else url
    return feed_map

# --- 記事データ読み込み関数 (変更なし) ---
@st.cache_data # DB読み込み結果をキャッシュする
def load_all_articles_from_db(conn_name=DB_CONNECTION_NAME, limit=INITIAL_ARTICLE_LOAD_LIMIT):
    """データベースから記事データを読み込み、リスト形式で返す (件数制限付き・キャッシュ対応)"""
    loaded_articles = []
    print(f"キャッシュ確認 or DB ({conn_name}) から最新 {limit} 件の記事データを読み込みます...")
    if not init_db(conn_name): st.error("データベースに接続できないため、記事を読み込めません。"); return []
    try:
        conn = st.connection(conn_name, type="sql", ttl=0)
        query = text("SELECT * FROM articles ORDER BY published_datetime_str DESC LIMIT :limit_val")
        df = conn.query(str(query), params={"limit_val": limit}, ttl=0)
        print(f"  - DBから {len(df)} 件の記事を取得しました (上限: {limit})。")
        loaded_articles = df.to_dict('records')
        for article_dict in loaded_articles:
            keywords_data = article_dict.get('keywords_tfidf')
            if isinstance(keywords_data, str):
                try: article_dict['keywords_tfidf'] = json.loads(keywords_data)
                except json.JSONDecodeError: article_dict['keywords_tfidf'] = []
            elif keywords_data is None: article_dict['keywords_tfidf'] = []
            article_dict['is_hidden'] = bool(article_dict.get('is_hidden', False))
            article_dict['is_liked'] = bool(article_dict.get('is_liked', False))
            article_dict['is_read'] = bool(article_dict.get('is_read', False))
    except sqlalchemy_exc.SQLAlchemyError as e: print(f"!!! SQLAlchemy DBエラー（読み込み）: {e} !!!"); st.error(f"データベースからの記事読み込み中にエラーが発生しました: {e}"); loaded_articles = []
    except Exception as e: print(f"!!! 予期せぬエラー（読み込み）: {e} !!!"); st.error(f"記事読み込み中に予期せぬエラーが発生しました: {e}"); traceback.print_exc(); loaded_articles = []
    print(f"データベースからのデータ読み込み・変換完了: {len(loaded_articles)} 件の記事")
    return loaded_articles

# --- ボタンアクション用ヘルパー関数 (変更なし) ---
def update_article_status(article_link_to_update, action, current_like_status=None, current_read_status=None, conn_name=DB_CONNECTION_NAME):
    """DBとSession State内の記事ステータスを更新する (DB読込キャッシュはクリアしない)"""
    print(f"\n--- DB & Session State の記事ステータス更新開始 ---")
    print(f"  アクション: {action}, 対象リンク: ...{article_link_to_update[-50:]}")
    sql_query = None
    params = {"link_val": article_link_to_update}
    new_status = None
    if action == 'toggle_like':
        if current_like_status is None: st.error("いいね状態更新エラー"); return False
        new_like_status = not current_like_status
        sql_query = text("UPDATE articles SET is_liked = :new_status WHERE link = :link_val")
        params["new_status"] = new_like_status
        new_status = {'is_liked': new_like_status}
        print(f"  - is_liked を {new_like_status} に更新します。")
    elif action == 'hide':
        sql_query = text("UPDATE articles SET is_hidden = true WHERE link = :link_val")
        params = {"link_val": article_link_to_update}
        new_status = {'is_hidden': True}
        print(f"  - is_hidden を true に更新します。")
    elif action == 'toggle_read':
        if current_read_status is None: st.error("既読状態更新エラー"); return False
        new_read_status = not current_read_status
        sql_query = text("UPDATE articles SET is_read = :new_status WHERE link = :link_val")
        params["new_status"] = new_read_status
        new_status = {'is_read': new_read_status}
        print(f"  - is_read を {new_read_status} に更新します。")
    else: st.error(f"不明なアクション: {action}"); return False
    db_updated = False
    try:
        conn = st.connection(conn_name, type="sql")
        with conn.session as s:
            result = s.execute(sql_query, params); s.commit()
            print(f"  - DB更新試行完了。影響を受けた行数: {result.rowcount}")
            if result.rowcount > 0: db_updated = True
            else: print(f"  - Warn: DB内で更新対象の記事が見つかりませんでした。")
    except sqlalchemy_exc.SQLAlchemyError as e: print(f"!!! SQLAlchemy DBエラー（更新）: {e} !!!"); st.error(f"記事状態の更新中にデータベースエラーが発生しました: {e}"); traceback.print_exc(); return False
    except Exception as e: print(f"!!! 予期せぬエラー（DB更新）: {e} !!!"); st.error(f"記事状態の更新中に予期せぬエラーが発生しました: {e}"); traceback.print_exc(); return False
    session_updated = False
    if SESSION_KEY_ARTICLES in st.session_state and isinstance(st.session_state[SESSION_KEY_ARTICLES], list):
        articles_in_state = st.session_state[SESSION_KEY_ARTICLES]
        for i, article in enumerate(articles_in_state):
            if article.get('link') == article_link_to_update:
                print(f"  - Session State 内の記事 (index: {i}) を更新します。")
                if new_status: articles_in_state[i].update(new_status); st.session_state[SESSION_KEY_LAST_INTERACTED] = article_link_to_update; session_updated = True; break
        if not session_updated: print(f"  - Warn: Session State 内で更新対象の記事が見つかりませんでした。")
    else: print("  - Warn: Session State に記事データが存在しないか、リスト形式ではありません。")
    if db_updated or session_updated: st.toast(f"記事の状態を更新しました。", icon="✅"); time.sleep(0.1); st.rerun(); return True
    else: st.warning("記事の状態更新に失敗しました（対象が見つからないか、DBエラー）。"); st.rerun(); return False

# --- 新規記事をDBに保存するヘルパー関数 (キャッシュクリア追加) ---
def insert_articles_to_db(articles_list, conn_name=DB_CONNECTION_NAME):
    """新しい記事のリストをデータベースに挿入し、関連キャッシュをクリアする"""
    if not articles_list: return 0
    print(f"\n--- {len(articles_list)} 件の新規記事を DB ({conn_name}) に挿入開始 ---")
    inserted_count = 0
    if not init_db(conn_name): st.error("データベースに接続できないため、新規記事を保存できません。"); return -1
    try:
        conn = st.connection(conn_name, type="sql")
        cols = ['link', 'title', 'published_datetime_str', 'summary', 'source_feed', 'id', 'summary_generated', 'keywords_tfidf', 'is_hidden', 'is_liked', 'is_read']
        sql_insert = text(f"INSERT INTO articles ({', '.join(cols)}) VALUES ({', '.join([f':{c}' for c in cols])}) ON CONFLICT (link) DO NOTHING")
        rows_to_insert = []
        for article_dict_raw in articles_list:
            params = {}
            for col in cols:
                if col == 'keywords_tfidf': params[col] = json.dumps(article_dict_raw.get(col, [])) if isinstance(article_dict_raw.get(col, []), list) and article_dict_raw.get(col, []) else None
                elif col in ['is_hidden', 'is_liked', 'is_read']: params[col] = False
                else: params[col] = article_dict_raw.get(col)
            rows_to_insert.append(params)
        with conn.session as s:
            try:
                result = s.execute(sql_insert, rows_to_insert)
                inserted_count = result.rowcount if result.rowcount >= 0 else len(articles_list)
                print(f"  - executemany による挿入試行完了。推定挿入/無視件数: {inserted_count}")
            except Exception as e_many:
                print(f"  - executemany 失敗 ({e_many})。1行ずつ挿入します...")
                inserted_count = 0
                for params in rows_to_insert:
                    try: result_single = s.execute(sql_insert, params); inserted_count += result_single.rowcount
                    except Exception as e_single: print(f"  - Warning: 行の挿入エラー (link: {params.get('link', 'N/A')[:50]}...): {e_single}")
            s.commit()
            print(f"  - DB挿入/無視処理完了。実際の挿入件数 (推定): {inserted_count}")
    except sqlalchemy_exc.SQLAlchemyError as e: print(f"!!! SQLAlchemy DBエラー（挿入）: {e} !!!"); st.error(f"新規記事保存DBエラー: {e}"); traceback.print_exc(); return -1
    except Exception as e: print(f"!!! 予期せぬエラー（挿入）: {e} !!!"); st.error(f"新規記事保存エラー: {e}"); traceback.print_exc(); return -1
    if inserted_count >= 0:
        st.session_state[SESSION_KEY_FORCE_REFRESH] = True; print("  - Session State の強制リフレッシュフラグを設定。")
        try: load_all_articles_from_db.clear(); print("  - load_all_articles_from_db キャッシュをクリア。")
        except Exception as e_clear: print(f"!!! DB読込キャッシュクリアエラー: {e_clear}")
        try: calculate_all_article_embeddings.clear(); print("  - calculate_all_article_embeddings キャッシュをクリア。")
        except Exception as e_clear: print(f"!!! Embeddingキャッシュクリアエラー: {e_clear}")
    return inserted_count

# --- 興味キーワードベクトル取得用キャッシュ関数 (変更なし) ---
@st.cache_data
def get_interest_vector(interest_keywords_tuple):
    """興味キーワードのタプルから結合テキストのEmbeddingベクトルを取得（キャッシュ対応）"""
    if not interest_keywords_tuple: return None
    if not IMPORT_SUCCESS or not GEMINI_INITIALIZED or get_embedding is None: return None
    interest_text = "\n".join(list(interest_keywords_tuple))
    print(f"  - 興味キーワードテキスト Embedding 取得（キャッシュ利用可）({len(interest_keywords_tuple)} keywords): {interest_text[:50]}...")
    try:
        interest_vector = get_embedding(interest_text, task_type="RETRIEVAL_QUERY")
        if interest_vector is None or not isinstance(interest_vector, list): print("  - Error: 興味キーワード Embedding 取得失敗。"); st.warning("興味キーワードベクトル化失敗。"); return None
        print(f"  - 興味キーワード Embedding 取得成功 (Dim: {len(interest_vector)})")
        return interest_vector
    except Exception as e: print(f"  - Error: 興味キーワード Embedding エラー: {e}"); st.error(f"興味キーワードベクトル化エラー: {e}"); traceback.print_exc(); return None

# --- 記事ベクトル計算＆キャッシュ用関数 (変更なし) ---
@st.cache_data
def calculate_all_article_embeddings(articles_json_tuple):
    """記事情報のJSON文字列タプルから、各記事のEmbeddingベクトル辞書を計算（キャッシュ対応）"""
    embeddings_dict = {}
    articles_list = []
    if not isinstance(articles_json_tuple, tuple): return embeddings_dict
    print(f"  - 記事 Embedding 計算開始（キャッシュ利用可）... 入力記事数: {len(articles_json_tuple)}")
    for article_json in articles_json_tuple:
        try: articles_list.append(json.loads(article_json))
        except json.JSONDecodeError as e: print(f"  - Warn: キャッシュJSONデコードエラー: {e}"); continue
    if not articles_list: return embeddings_dict
    if not IMPORT_SUCCESS or not GEMINI_INITIALIZED or get_embedding is None: return embeddings_dict
    print(f"  - {len(articles_list)} 件の記事 Embedding 計算実行...")
    texts_to_embed = []; links_in_order = []
    for article in articles_list:
        link = article.get('link'); title = article.get('title', ''); summary = article.get('summary_generated', article.get('summary', ''))
        text = f"記事タイトル: {title}\n記事要約: {summary}".strip();
        if link and text: texts_to_embed.append(text); links_in_order.append(link)
    if not texts_to_embed: return embeddings_dict
    try:
        print(f"  - Gemini Embedding API 呼び出し ({len(texts_to_embed)} texts)...")
        embeddings_list = get_embedding(texts_to_embed, task_type="RETRIEVAL_DOCUMENT")
        if embeddings_list and len(embeddings_list) == len(links_in_order):
            valid_embedding_count = 0
            for link, vector in zip(links_in_order, embeddings_list):
                if vector and isinstance(vector, list): embeddings_dict[link] = vector; valid_embedding_count += 1
            print(f"  - Embedding計算成功: {valid_embedding_count} 件取得。")
        elif embeddings_list: print(f"  - Error: Embedding結果数不一致"); st.warning("記事ベクトル一部取得失敗。")
        else: print("  - Error: Embedding取得失敗"); st.error("記事ベクトル取得失敗。")
    except Exception as e: print(f"  - Error: Embedding 計算エラー: {e}"); st.error(f"記事ベクトル計算エラー: {e}"); traceback.print_exc()
    return embeddings_dict

# --- 類似度計算関数 (変更なし) ---
def calculate_recommendation_scores(interest_keywords_list, article_embeddings_dict):
    """興味キーワードリストと記事ベクトル辞書から推薦スコアを計算"""
    scores_dict = {}
    if not interest_keywords_list or not article_embeddings_dict: return scores_dict
    print(f"興味キーワード ({len(interest_keywords_list)}個) に基づく推薦スコア計算...")
    interest_keywords_tuple = tuple(sorted(interest_keywords_list))
    interest_vector = get_interest_vector(interest_keywords_tuple)
    if interest_vector is None: return scores_dict
    article_links = []; article_vectors = []
    for link, vector in article_embeddings_dict.items():
        if vector and isinstance(vector, list): article_links.append(link); article_vectors.append(vector)
    if not article_vectors: return scores_dict
    try:
        interest_vector_np = np.array(interest_vector).reshape(1, -1)
        article_vectors_np = np.array(article_vectors)
        if interest_vector_np.shape[1] != article_vectors_np.shape[1]: print(f"  - Error: Vector dimension mismatch!"); st.error("ベクトル次元不一致"); return scores_dict
        print(f"  - {len(article_vectors_np)} 件の記事ベクトルとのコサイン類似度計算...")
        similarity_scores = cosine_similarity(interest_vector_np, article_vectors_np)
        scores_list = similarity_scores[0]; print(f"  - 類似度計算成功。")
        for link, score in zip(article_links, scores_list): scores_dict[link] = float(score)
    except ValueError as e_np: print(f"  - Error: NumPy配列処理エラー: {e_np}"); st.error(f"ベクトル処理エラー: {e_np}"); traceback.print_exc(); return {}
    except Exception as e_sim: print(f"  - Error: 類似度計算エラー: {e_sim}"); st.error(f"類似度計算エラー: {e_sim}"); traceback.print_exc(); return {}
    return scores_dict

# --- いいねブーストスコア計算関数 (変更なし) ---
def calculate_liked_boost_scores(liked_links_list, all_article_embeddings):
    """「いいね」記事に基づいてブーストスコアを計算"""
    boost_scores = {}
    if not liked_links_list or not all_article_embeddings: return boost_scores
    print(f"\n--- {len(liked_links_list)} 件の「いいね」記事に基づいてブーストスコア計算 ---")
    liked_vectors = []
    for link in liked_links_list:
        vector = all_article_embeddings.get(link)
        if vector and isinstance(vector, list): liked_vectors.append(vector)
    if not liked_vectors: return boost_scores
    try:
        avg_liked_vector_np = np.mean(np.array(liked_vectors), axis=0)
        avg_liked_vector = avg_liked_vector_np.reshape(1, -1)
        print(f"  - 平均いいねベクトル計算成功 (Shape: {avg_liked_vector.shape})")
    except Exception as e: print(f"  - Error: 平均いいねベクトル計算エラー: {e}"); st.error(f"いいね平均ベクトル計算エラー: {e}"); return boost_scores
    target_links = []; target_vectors = []
    for link, vector in all_article_embeddings.items():
        if vector and isinstance(vector, list): target_links.append(link); target_vectors.append(vector)
    if not target_vectors: return boost_scores
    try:
        target_vectors_np = np.array(target_vectors)
        if avg_liked_vector.shape[1] != target_vectors_np.shape[1]: print(f"  - Error: Dimension mismatch (avg_liked vs targets)"); st.error("ベクトル次元不一致(Boost)"); return {}
        print(f"  - 平均いいねベクトルと {len(target_vectors_np)} 件の記事ベクトルとの類似度計算...")
        similarity_scores = cosine_similarity(avg_liked_vector, target_vectors_np)
        scores_list = similarity_scores[0]; print(f"  - ブーストスコア計算成功。")
        for link, score in zip(target_links, scores_list): boost_scores[link] = float(score)
    except ValueError as e_np: print(f"  - Error: NumPy配列処理エラー(Boost): {e_np}"); st.error(f"ベクトル処理エラー(Boost): {e_np}"); traceback.print_exc(); return {}
    except Exception as e_sim: print(f"  - Error: ブーストスコア類似度計算エラー: {e_sim}"); st.error(f"類似度計算エラー(Boost): {e_sim}"); traceback.print_exc(); return {}
    return boost_scores

# --- Gemini 推薦理由生成関数 (モデル名変更) ---
@st.cache_data
def get_recommendation_reason(_article_link, article_title, article_summary, article_keywords_tuple, interest_keywords_tuple):
    """Gemini API を使用して推薦理由を生成（キャッシュ対応）"""
    if not GEMINI_INITIALIZED or not interest_keywords_tuple: return None
    interest_keywords = list(interest_keywords_tuple); article_keywords = list(article_keywords_tuple)
    print(f"  - 推薦理由生成（キャッシュ利用可）... 対象記事: {article_title[:30]}...")
    prompt = f"""ユーザーは以下のキーワードに興味を持っています: {', '.join(interest_keywords)}\n\n以下の記事について、上記のユーザーの興味とどのように関連しているか、推薦する理由を1～2文で具体的に、かつ簡潔に説明してください。\n\n記事タイトル: {article_title}\n記事要約: {article_summary}\n記事キーワード: {', '.join(article_keywords)}\n\n推薦理由："""
    try:
        model_name = 'gemini-2.0-flash-lite' # ユーザー指定のモデル名に変更
        print(f"  - Using Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        reason_text = response.text.strip()
        print(f"    -> 推薦理由生成成功。理由: {reason_text[:50]}...")
        return reason_text
    except Exception as e:
        print(f"    - Error: Gemini API 推薦理由生成エラー (Model: {model_name}): {e}")
        st.warning(f"推薦理由の生成に失敗しました (モデル: {model_name})。モデル名が正しいか確認してください。")
        print("\n--- Traceback (Gemini API Error) ---"); traceback.print_exc(); print("--- End Traceback ---")
        return None

# --- 記事表示用関数 (アイコン変更、ボタンレイアウト修正) ---
def display_article(article_data, key_prefix, feed_map, show_reason=False, interest_keywords_list=None):
    """記事データを整形して Streamlit コンテナ内に表示する"""
    article_link = article_data.get('link')
    if not article_link: return
    is_liked_current = article_data.get('is_liked', False)
    is_read_current = article_data.get('is_read', False)
    title = article_data.get('title', 'タイトルなし')
    kw_list = article_data.get('keywords_tfidf', [])
    summary = article_data.get('summary_generated', article_data.get('summary', ''))
    date_str = article_data.get('published_datetime_str', None)
    source_url = article_data.get('source_feed', '')
    source_name = feed_map.get(source_url, source_url)
    with st.container(border=True):
        col_date, col_main, col_meta = st.columns([1, 5, 2])
        with col_date: # 日付
            date_disp = "--"
            if date_str:
                try: dt_obj = pd.to_datetime(date_str.replace(' GMT',''), errors='coerce'); date_disp = dt_obj.strftime('%m/%d') if pd.notna(dt_obj) else "--"
                except Exception as e_date: print(f"日付変換エラー: {e_date}")
            st.caption(f"{date_disp}")
        with col_main: # タイトル、要約、理由
            title_display = f"**{title}**"; title_display = f"~~{title_display}~~" if is_read_current else title_display
            st.markdown(f"[{title_display}]({article_link})", unsafe_allow_html=True)
            if summary: summary_style = "font-size: smaller;" + (" opacity: 0.7;" if is_read_current else ""); st.markdown(f'<span style="{summary_style}">{summary}</span>', unsafe_allow_html=True)
            if show_reason and GEMINI_INITIALIZED and interest_keywords_list:
                with st.spinner("🤖 おすすめ理由..."):
                    interest_keywords_tuple = tuple(sorted(interest_keywords_list)); article_keywords_tuple = tuple(sorted(kw_list))
                    reason = get_recommendation_reason(_article_link=article_link, article_title=title, article_summary=summary, article_keywords_tuple=article_keywords_tuple, interest_keywords_tuple=interest_keywords_tuple)
                if reason: st.markdown("---"); st.markdown(f"💡 **理由:** {reason}")
        with col_meta: # ソース、キーワード、ボタン
            if source_name: st.caption(f"{source_name}")
            if kw_list: kw_tags = [f"`{k}`" for k in kw_list]; st.markdown(f"<small>{' '.join(kw_tags)}</small>", unsafe_allow_html=True)
            st.markdown("---")
            # ★★★ ボタン縦積み許容レイアウト ★★★
            like_icon = "❤️" if is_liked_current else "🤍"
            if st.button(f"{like_icon}", key=f"{key_prefix}_like_{article_link}", help="いいね/解除", use_container_width=True): update_article_status(article_link, 'toggle_like', current_like_status=is_liked_current)
            if st.button("🗑️", key=f"{key_prefix}_hide_{article_link}", help="非表示", use_container_width=True): update_article_status(article_link, 'hide')
            read_icon = "✔️" if is_read_current else "📘" # ★アイコン変更
            read_help = "未読にする" if is_read_current else "既読にする"
            if st.button(read_icon, key=f"{key_prefix}_read_{article_link}", help=read_help, use_container_width=True): update_article_status(article_link, 'toggle_read', current_read_status=is_read_current)

# --- Streamlit アプリケーション 本体 ---
# ★★★ タイトル変更 ★★★
st.title("📰 Signal Spotter")

# --- 初期化チェック ---
if not IMPORT_SUCCESS: st.error(f"アプリケーションの起動に必要なモジュール({IMPORT_ERROR_MESSAGE})を読み込めませんでした。"); st.stop()
db_available = init_db(DB_CONNECTION_NAME)
if not db_available: st.warning("データベースに接続できません。記事関連機能は利用できません。")

# --- データロード & Session State 管理 ---
# ★★★ DBからフィード情報をロード ★★★
print(f"--- Calling load_feeds_from_db ---") # ★デバッグプリント変更
loaded_feed_data = []
if db_available: loaded_feed_data = load_feeds_from_db() # ★バージョン引数削除
feed_map_for_display = get_feed_map_from_list(loaded_feed_data)

# ★★★ DBから記事情報をロード (Session State管理) ★★★
if SESSION_KEY_ARTICLES not in st.session_state or st.session_state.get(SESSION_KEY_FORCE_REFRESH, False):
    if db_available:
        st.session_state[SESSION_KEY_ARTICLES] = load_all_articles_from_db(DB_CONNECTION_NAME, limit=INITIAL_ARTICLE_LOAD_LIMIT)
        st.session_state[SESSION_KEY_FORCE_REFRESH] = False
        print(f"記事データをDBから(or キャッシュから)最大 {INITIAL_ARTICLE_LOAD_LIMIT} 件読み込み、Session Stateに格納しました。")
    else: st.session_state[SESSION_KEY_ARTICLES] = []; print("DB接続不可のため、Session Stateの記事データを空にしました。")
articles_data = st.session_state.get(SESSION_KEY_ARTICLES, [])

# --- 記事ベクトル計算 (記事データが存在する場合) ---
article_embeddings = {}
if articles_data and IMPORT_SUCCESS and GEMINI_INITIALIZED:
    try:
        articles_tuple_for_cache = tuple(json.dumps({k: v for k, v in a.items() if k not in ['is_hidden', 'is_liked', 'is_read']}, sort_keys=True, ensure_ascii=False) for a in articles_data)
        print(f"記事ベクトル計算のためのキャッシュキー（タプル）を作成 (要素数: {len(articles_tuple_for_cache)})")
        article_embeddings = calculate_all_article_embeddings(articles_tuple_for_cache)
    except Exception as e_cache: st.error(f"記事ベクトルの計算/キャッシュ処理中にエラー: {e_cache}"); print(f"!!! ベクトル計算/キャッシュ処理エラー: {e_cache}"); traceback.print_exc(); article_embeddings = {}
elif not (IMPORT_SUCCESS and GEMINI_INITIALIZED): st.warning("Gemini Embedding 機能が利用できないため、記事ベクトルは計算されません。")

# --- サイドバー ---
with st.sidebar:
    st.header("🔍 検索 & 絞り込み")
    search_keyword = st.text_input("要約内容で検索:", key=SESSION_KEY_SEARCH_BOX, value=st.session_state.get(SESSION_KEY_SEARCH_BOX, ""))

    st.markdown("---"); st.markdown("#### ℹ️ 情報源フィルタ")
    all_source_names = []; name_to_url_map = {}
    # ★★★ DBからロードしたフィードデータを使用 ★★★
    if loaded_feed_data:
        all_source_names = sorted([item.get('name', item.get('url')) for item in loaded_feed_data])
        name_to_url_map = {item.get('name', item.get('url')): item.get('url') for item in loaded_feed_data}
    if all_source_names:
        selected_source_names = st.multiselect("情報源で絞り込み:", options=all_source_names, default=st.session_state.get(SESSION_KEY_SOURCE_FILTER, []), key=SESSION_KEY_SOURCE_FILTER)
    else: st.caption("利用可能な情報源なし")

    # --- フィード管理 ---
    st.markdown("---"); st.header("⚙️ フィード管理"); st.markdown("##### 現在のフィード")
    # ★★★ DBからロードしたフィードデータを使用 ★★★
    if loaded_feed_data:
        feeds_df = pd.DataFrame(loaded_feed_data)
        st.dataframe(feeds_df, hide_index=True, use_container_width=True, column_config={"url": st.column_config.LinkColumn("URL", display_text="🔗")})
        st.markdown("##### フィード削除")
        delete_options = [""] + all_source_names # 名前リストを再利用
        feed_name_to_delete = st.selectbox("削除するフィードを選択:", options=delete_options, index=0, key="delete_feed_select")
        if feed_name_to_delete:
            if st.button(f"「{feed_name_to_delete}」を削除", key=f"delete_button_{feed_name_to_delete}", type="primary"):
                url_to_delete = name_to_url_map.get(feed_name_to_delete)
                if url_to_delete:
                    # ★★★ DBからフィードを削除 ★★★
                    if delete_feed_from_db(url_to_delete):
                        st.success(f"フィード「{feed_name_to_delete}」削除成功"); time.sleep(1); st.rerun()
                    else: st.error("フィード削除失敗")
                else: st.error(f"削除対象URLが見つかりません: {feed_name_to_delete}")
    else: st.caption("登録フィードなし")
    st.markdown("##### 新規フィード追加")
    with st.form("add_feed_form", clear_on_submit=True):
        new_feed_name = st.text_input("表示名", placeholder="例: GIGAZINE"); new_feed_url = st.text_input("RSS/AtomフィードURL", placeholder="https://...")
        submitted = st.form_submit_button("フィードを追加")
        if submitted:
            is_valid_url = new_feed_url and (new_feed_url.startswith('http://') or new_feed_url.startswith('https://'))
            if new_feed_name and is_valid_url:
                # ★★★ DBにフィードを追加 ★★★
                if add_feed_to_db(new_feed_name, new_feed_url):
                    st.success(f"フィード '{new_feed_name}' 追加成功"); time.sleep(1); st.rerun()
                else: st.error("フィード追加失敗（既に登録済みかDBエラー）") # エラーメッセージを少し変更
            elif not new_feed_name: st.warning("表示名を入力してください。")
            else: st.warning("有効なフィードURLを入力してください。")

    # --- 興味キーワード管理 ---
    st.markdown("---"); st.header("💡 興味キーワード"); st.caption("レコメンデーションに使用。(改行区切り)")
    # ★★★ DBからキーワードをロード (Session State経由) ★★★
    if SESSION_KEY_INTEREST_KEYWORDS not in st.session_state:
        if db_available: st.session_state[SESSION_KEY_INTEREST_KEYWORDS] = load_keywords_from_db() # ★バージョン引数削除
        else: st.session_state[SESSION_KEY_INTEREST_KEYWORDS] = [] # DBなければ空

    with st.form("interest_form", clear_on_submit=False):
        interest_input = st.text_area("キーワード:", value="\n".join(st.session_state.get(SESSION_KEY_INTEREST_KEYWORDS, [])), key="interest_keyword_input_in_form", height=150)
        submitted_interest = st.form_submit_button("興味キーワードを更新")
        if submitted_interest:
            keywords_raw = interest_input.splitlines()
            updated_keywords = sorted(list(set(kw.strip() for kw in keywords_raw if kw.strip())))
            # ★★★ DBにキーワードを保存 ★★★
            if save_keywords_to_db(updated_keywords):
                st.session_state[SESSION_KEY_INTEREST_KEYWORDS] = updated_keywords # Session Stateも更新
                st.success(f"興味キーワード ({len(updated_keywords)}個) 更新成功"); st.rerun()
            else: st.error("キーワード保存失敗")
    if st.session_state.get(SESSION_KEY_INTEREST_KEYWORDS):
        st.write("現在の興味キーワード:"); tags_md = [f"`{kw}`" for kw in st.session_state[SESSION_KEY_INTEREST_KEYWORDS]]; st.markdown(" ".join(tags_md))
    else: st.caption("興味キーワード未登録")
    if db_available and article_embeddings: st.success(f"{len(article_embeddings)}件記事ベクトル準備完了", icon="✅")
    elif db_available: st.info("記事ベクトル未計算")

# --- メインエリア ---
st.markdown("---")

# --- 更新ボタン ---
if IMPORT_SUCCESS and run_curation_pipeline and db_available:
    if st.button("🔄 新しい記事をチェック＆DB更新", key="update_button", use_container_width=True):
        with st.spinner("新しい記事を取得・処理中です..."):
            try:
                # ★★★ DBから読み込んだフィードリストを渡す ★★★
                current_feeds = load_feeds_from_db() # ★バージョン引数削除
                if not current_feeds:
                     st.warning("収集対象のフィードが登録されていません。サイドバーから追加してください。")
                else:
                    # ★★★ run_curation_pipeline 呼び出し修正 ★★★
                    new_unique_articles = run_curation_pipeline(feed_list=current_feeds) # 引数で渡す
                    if new_unique_articles is not None:
                        if new_unique_articles:
                            print(f"パイプラインから {len(new_unique_articles)} 件の新規記事候補を取得。")
                            inserted_count = insert_articles_to_db(new_unique_articles, DB_CONNECTION_NAME)
                            if inserted_count >= 0: st.success(f"更新処理完了。DBに {inserted_count} 件追加(or無視)。表示更新。"); time.sleep(1); st.rerun()
                            else: st.error("記事のDB保存エラー。")
                        else: st.info("新しい記事は見つかりませんでした。")
                    else: st.error("記事の取得・処理エラー。")
            except TypeError as e_type: # ★★★ TypeError を捕捉 ★★★
                print(f"!!! TypeError in run_curation_pipeline call: {e_type}")
                traceback.print_exc()
                st.error(f"記事更新処理の呼び出しでエラーが発生しました。`curation_logic.py` が `feed_list` 引数を受け取るように修正されているか確認してください。 Error: {e_type}")
            except Exception as e_pipeline: print(f"!!! run_curation_pipeline エラー: {e_pipeline}"); traceback.print_exc(); st.error(f"記事更新パイプライン実行エラー: {e_pipeline}")
elif not db_available: st.info("DBに接続できないため、記事の更新はできません。")
else: st.warning("`curation_logic.py` 未検出のため、記事更新機能は利用不可。")

# --- 表示用記事リストの準備 (Session Stateから) ---
visible_articles_list = []
if articles_data:
    visible_articles_list = [article for article in articles_data if not article.get('is_hidden', False)]
    print(f"Session State から取得した非表示除外記事数: {len(visible_articles_list)}")
else: print("Session State に表示可能な記事データがありません。")

# --- おすすめ記事表示 (日付フィルター適用) ---
st.subheader("⭐ あなたへのおすすめ記事")
user_keywords = st.session_state.get(SESSION_KEY_INTEREST_KEYWORDS, [])
can_recommend = (user_keywords and article_embeddings and visible_articles_list and IMPORT_SUCCESS and GEMINI_INITIALIZED)
if can_recommend:
    unread_articles_for_rec = [a for a in visible_articles_list if not a.get('is_read', False)]
    print(f"レコメンデーション対象の未読記事数: {len(unread_articles_for_rec)}")
    if unread_articles_for_rec:
        base_scores = calculate_recommendation_scores(user_keywords, article_embeddings)
        liked_links = [a['link'] for a in articles_data if a.get('is_liked') and a.get('link')]
        boost_scores = calculate_liked_boost_scores(liked_links, article_embeddings)
        final_scores = {}
        for article in unread_articles_for_rec:
            link = article.get('link')
            if link: base_score = base_scores.get(link, 0.0); boost_score = boost_scores.get(link, 0.0); final_scores[link] = base_score + (RECOMMENDATION_BOOST_WEIGHT * boost_score)
        if final_scores:
            sorted_final_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
            recommended_links = [link for link, score in sorted_final_scores if score >= RECOMMENDATION_THRESHOLD_SCORE][:RECOMMENDATION_COUNT]
            if recommended_links:
                articles_dict = {article.get('link'): article for article in unread_articles_for_rec if article.get('link')}
                cutoff_duration = pd.Timedelta(days=3); now_utc = pd.Timestamp.now(tz='UTC'); cutoff_time = now_utc - cutoff_duration
                filtered_recommended_links = []
                print(f"Applying 3-day filter to {len(recommended_links)} recommended links (cutoff: {cutoff_time})...")
                for link in recommended_links:
                    article_data = articles_dict.get(link); keep_article = False
                    if article_data:
                        date_str = article_data.get('published_datetime_str')
                        if date_str:
                            try:
                                pub_date_utc = pd.to_datetime(date_str.replace(' GMT',''), errors='coerce', utc=True)
                                if pd.notna(pub_date_utc):
                                    if pub_date_utc >= cutoff_time: keep_article = True
                                    else: print(f"  - Filtering out old article (recommendation): {link} (Published: {pub_date_utc})")
                                else: print(f"  - Filtering out article with unparseable date (recommendation): {link} (DateStr: {date_str})")
                            except Exception as e_date: print(f"  - Error parsing date for filtering recommended article {link}: {e_date}. Filtering out.")
                        else: print(f"  - Filtering out article with missing date (recommendation): {link}")
                    else: print(f"  - Warn: Could not find article data for recommended link {link}. Filtering out.")
                    if keep_article: filtered_recommended_links.append(link)
                print(f" -> Filtered recommended links: {len(filtered_recommended_links)}件")
                recommended_articles_info = [articles_dict.get(link) for link in filtered_recommended_links if articles_dict.get(link)]
                if recommended_articles_info:
                    st.caption(f"興味キーワード (+いいね) に基づくおすすめ上位 {len(recommended_articles_info)} 件 (公開3日以内, スコア >= {RECOMMENDATION_THRESHOLD_SCORE}):")
                    for article_data in recommended_articles_info: display_article(article_data, key_prefix="rec", feed_map=feed_map_for_display, show_reason=True, interest_keywords_list=user_keywords)
                else: st.info("おすすめ記事はありますが、公開から3日以内のものはありませんでした。")
            else: st.info(f"スコア {RECOMMENDATION_THRESHOLD_SCORE} 以上のおすすめ記事はありませんでした。")
        else: st.info("おすすめスコアを計算できませんでした。")
    else: st.info("未読の記事がないため、おすすめを表示できません。")
elif not user_keywords: st.info("サイドバーで興味キーワードを登録すると、おすすめ記事が表示されます。")
elif not article_embeddings: st.warning("記事ベクトル未計算のため、おすすめ機能は利用不可。")
elif not visible_articles_list: st.info("表示可能な記事がありません。")
else: st.warning("レコメンデーション機能に必要な設定（Gemini API等）が不足している可能性あり。")

# --- 全記事一覧表示 ---
st.markdown("---"); st.subheader("📰 収集済み記事一覧")

# フィルター処理 (Session State のデータに対して行う)
filtered_articles = visible_articles_list
current_search_keyword = st.session_state.get(SESSION_KEY_SEARCH_BOX, '')
if current_search_keyword:
    print(f"検索キーワード「{current_search_keyword}」でフィルタリング...")
    try: filtered_articles = [a for a in filtered_articles if current_search_keyword.lower() in str(a.get('summary_generated', '')).lower()]; print(f" -> 検索結果: {len(filtered_articles)} 件")
    except Exception as e_search: print(f"!!! 検索フィルタエラー: {e_search}"); st.error(f"検索エラー: {e_search}")

current_selected_sources = st.session_state.get(SESSION_KEY_SOURCE_FILTER, [])
if current_selected_sources:
    print(f"選択された情報源 {current_selected_sources} でフィルタリング...")
    urls_to_filter = [name_to_url_map.get(name) for name in current_selected_sources if name_to_url_map.get(name)]
    if urls_to_filter:
        try: filtered_articles = [a for a in filtered_articles if a.get('source_feed') in urls_to_filter]; print(f" -> 情報源フィルタ結果: {len(filtered_articles)} 件")
        except Exception as e_filter: print(f"!!! 情報源フィルタエラー: {e_filter}"); st.error(f"情報源フィルタエラー: {e_filter}")
    else: print(" -> 選択された情報源に対応するURLなし。")

# フィルタリング結果表示
if filtered_articles:
    st.caption(f"表示件数: {len(filtered_articles)} 件 (最大 {INITIAL_ARTICLE_LOAD_LIMIT} 件中のフィルタ結果)")
    try: # 日付ソート
        def get_datetime_safe(article):
            date_str = article.get('published_datetime_str'); dt = pd.Timestamp.min
            if date_str:
                try: dt_parsed = pd.to_datetime(date_str.replace(' GMT',''), errors='coerce'); dt = dt_parsed if pd.notna(dt_parsed) else dt
                except Exception as e_parse: print(f"  - Warn: 日付パースエラー for '{date_str}': {e_parse}")
            return dt
        sorted_articles = sorted(filtered_articles, key=get_datetime_safe, reverse=True)
    except Exception as e_sort: print(f"!!! 日付ソート中にエラー: {e_sort}"); st.warning("記事の日付ソート中にエラーが発生しました。元の順序で表示します。"); sorted_articles = filtered_articles

    for article_data in sorted_articles: # ソート済みリストを表示
        display_article(article_data, key_prefix="all", feed_map=feed_map_for_display, show_reason=False)

elif visible_articles_list: st.info("指定された検索・絞り込み条件に一致する記事は見つかりませんでした。")
elif not articles_data and db_available: st.info("表示可能な記事がありません。「新しい記事をチェック＆DB更新」ボタンで記事を取得してください。")
else: st.info("表示する記事がありません。")

# --- フッター ---
st.markdown("---"); st.caption("Signal Spotter (Settings DB Storage)")

