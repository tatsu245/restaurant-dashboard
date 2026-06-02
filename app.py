import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import anthropic
import json

st.set_page_config(
    page_title="店舗別時間帯売上ダッシュボード",
    page_icon="🍽️",
    layout="wide",
)

HOURS = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0,1,2,3,4,5]
HOUR_COLS = [f"{h}時" for h in HOURS]
WEEKDAY_MAP = {0:"月",1:"火",2:"水",3:"木",4:"金",5:"土",6:"日"}
WEEKDAY_ORDER = ["月","火","水","木","金","土","日"]

# ── CSV読み込み ───────────────────────────────────────
def load_from_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    # Shift-JIS → UTF-8 フォールバック
    for enc in ("shift-jis", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("ファイルのエンコードを判別できませんでした（Shift-JIS / UTF-8 を試してください）。")
        return pd.DataFrame()

    lines = text.splitlines()
    data_lines = []
    for line in lines[3:]:   # 先頭3行はヘッダー行
        parts = line.split(",")
        if len(parts) < 5:
            continue
        if not parts[0] or parts[2] in ("合計", ""):
            continue
        data_lines.append(parts[:29])

    if not data_lines:
        return pd.DataFrame()

    header = ["store_id", "store_name", "date", "type"] + HOUR_COLS + ["total"]
    df = pd.DataFrame(data_lines, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df = df[df["date"].notna()].copy()
    for c in HOUR_COLS + ["total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    df["weekday"] = df["date"].dt.weekday.map(WEEKDAY_MAP)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["store_label"] = df["store_id"].astype(str) + " " + df["store_name"]
    return df

def split_types(df):
    return df[df["type"] == "売上"].copy(), df[df["type"] == "客数"].copy()

# ── AI分析 ───────────────────────────────────────────
def build_analysis_prompt(sales_df, cust_df) -> str:
    store_summary = (
        sales_df.groupby("store_name")["total"].sum()
        .sort_values(ascending=False)
        .apply(lambda x: f"¥{x:,.0f}")
        .to_dict()
    )
    cust_summary = (
        cust_df.groupby("store_name")["total"].sum()
        .sort_values(ascending=False)
        .apply(lambda x: f"{x:,.0f}人")
        .to_dict()
    )
    hourly_sales = sales_df[HOUR_COLS].sum().apply(lambda x: f"¥{x:,.0f}").to_dict()
    hourly_cust  = cust_df[HOUR_COLS].sum().apply(lambda x: f"{x:,.0f}人").to_dict()

    s_hour = sales_df[HOUR_COLS].sum()
    c_hour = cust_df[HOUR_COLS].sum()
    unit_by_hour = (s_hour / c_hour.replace(0, float("nan"))).fillna(0).apply(lambda x: f"¥{x:,.0f}").to_dict()

    weekday_sales = (
        sales_df.groupby("weekday")["total"].sum()
        .reindex(WEEKDAY_ORDER)
        .apply(lambda x: f"¥{x:,.0f}")
        .to_dict()
    )
    monthly_sales = (
        sales_df.groupby(["month", "store_name"])["total"].sum()
        .unstack(fill_value=0)
        .to_dict()
    )
    months = sorted(sales_df["month"].unique())
    monthly_str = []
    for m in months:
        row = {k: f"¥{v:,.0f}" for k, v in monthly_sales.get(m, {}).items()}
        monthly_str.append(f"{m}: {row}")

    idle_hours = (sales_df[HOUR_COLS] == 0).mean().sort_values(ascending=False).head(5)
    idle_str = {k: f"{v*100:.0f}%" for k, v in idle_hours.items()}

    return f"""あなたは飲食業界の経営コンサルタントです。以下のデータをもとに、経営者・マーケターが次のアクションを決めるための示唆・分析を日本語で出力してください。

## データ概要
- 分析期間：{sales_df['date'].min().strftime('%Y/%m/%d')} 〜 {sales_df['date'].max().strftime('%Y/%m/%d')}
- 対象店舗数：{sales_df['store_name'].nunique()}店舗

## 店舗別 売上合計（期間内）
{json.dumps(store_summary, ensure_ascii=False, indent=2)}

## 店舗別 客数合計
{json.dumps(cust_summary, ensure_ascii=False, indent=2)}

## 時間帯別 売上合計（全店舗）
{json.dumps(hourly_sales, ensure_ascii=False, indent=2)}

## 時間帯別 客単価
{json.dumps(unit_by_hour, ensure_ascii=False, indent=2)}

## 曜日別 売上合計
{json.dumps(weekday_sales, ensure_ascii=False, indent=2)}

## 月別 店舗別 売上推移
{chr(10).join(monthly_str)}

## アイドルタイム（売上¥0の割合が高い時間帯TOP5）
{json.dumps(idle_str, ensure_ascii=False, indent=2)}

---

以下の5点について、それぞれ具体的な数値を引用しながら分析してください：

1. **パフォーマンス評価**：売上・客数・客単価の観点から店舗を3段階（高/中/低）で分類し、特徴を述べる
2. **ピーク・アイドル時間帯**：稼ぎ頭の時間帯と集客が薄い時間帯を特定し、その意味を述べる
3. **月別トレンド**：2月〜4月の変化で注目すべき動きがある店舗や傾向を述べる
4. **曜日特性**：平日・週末の差から読み取れる客層・行動特性を述べる
5. **施策提案**：アイドルタイム解消・客単価向上・集客強化に向けた具体的な施策を3〜5個提案する（SNS/LINE/タイムセール等も含めて）
"""

def run_ai_analysis(sales_df, cust_df) -> str:
    try:
        api_key = st.secrets.get("anthropic", {}).get("api_key", "")
    except Exception:
        api_key = ""
    if not api_key:
        return "💡 AI分析はStreamlit Cloudデプロイ後にご利用いただけます。"
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_analysis_prompt(sales_df, cust_df)
    with st.spinner("Claude が分析中..."):
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
    return message.content[0].text

# ── サイドバー ──────────────────────────────────────
st.sidebar.title("🔧 設定")

uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=["csv"],
    help="時間帯別売上CSVファイル（Shift-JIS）をドラッグ＆ドロップしてください",
)

if st.sidebar.button("🔄 データをクリア"):
    st.session_state.pop("df_all", None)
    st.rerun()

st.sidebar.divider()

# ── データ読み込み ───────────────────────────────────
if uploaded_file is not None:
    with st.spinner("CSVを読み込み中..."):
        df_all = load_from_csv(uploaded_file)
    if not df_all.empty:
        st.session_state["df_all"] = df_all
        st.session_state["filename"] = uploaded_file.name

if "df_all" not in st.session_state:
    st.title("🍽️ 店舗別時間帯売上ダッシュボード")
    st.info("👈 サイドバーからCSVファイルをアップロードしてください。")
    st.caption("対応形式：時間帯別売上CSV（Shift-JIS）")
    st.stop()

df_all = st.session_state["df_all"]
filename = st.session_state.get("filename", "")

# ── フィルター ───────────────────────────────────────
stores = sorted(df_all["store_label"].unique())
selected_stores = st.sidebar.multiselect("店舗を選択", stores, default=stores)

months = sorted(df_all["month"].unique())
selected_months = st.sidebar.multiselect("月を選択", months, default=months)

if filename:
    st.sidebar.caption(f"📄 {filename}")

df = df_all[
    df_all["store_label"].isin(selected_stores) &
    df_all["month"].isin(selected_months)
].copy()

if df.empty:
    st.warning("フィルター条件にデータがありません。")
    st.stop()

sales_df, cust_df = split_types(df)

# ── KPIカード ────────────────────────────────────────
st.title("🍽️ 店舗別時間帯売上ダッシュボード")

total_sales = int(sales_df["total"].sum())
total_cust  = int(cust_df["total"].sum())
unit_price  = int(total_sales / total_cust) if total_cust > 0 else 0
peak_hour   = sales_df[HOUR_COLS].sum().idxmax()

k1, k2, k3, k4 = st.columns(4)
k1.metric("総売上", f"¥{total_sales:,}")
k2.metric("総客数", f"{total_cust:,} 人")
k3.metric("平均客単価", f"¥{unit_price:,}")
k4.metric("最多売上時間", peak_hour)

st.divider()

# ── AI分析 ───────────────────────────────────────────
with st.expander("🤖 AIによる示唆・分析", expanded=False):
    if st.button("Claude に分析させる", type="primary"):
        result = run_ai_analysis(sales_df, cust_df)
        st.session_state["ai_result"] = result
    if "ai_result" in st.session_state:
        st.markdown(st.session_state["ai_result"])

st.divider()

# ── ヒートマップ ──────────────────────────────────────
st.subheader("⏰ 時間帯 × 曜日 売上ヒートマップ")

hm = (
    sales_df.groupby("weekday")[HOUR_COLS].sum()
    .reindex(WEEKDAY_ORDER)
    .fillna(0)
)
fig_hm = px.imshow(
    hm,
    labels=dict(x="時間帯", y="曜日", color="売上 (円)"),
    color_continuous_scale="Blues",
    aspect="auto",
)
fig_hm.update_layout(height=320, margin=dict(t=20, b=20))
st.plotly_chart(fig_hm, use_container_width=True)

st.divider()

# ── 月別推移 & 店舗ランキング ────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 月別売上推移")
    monthly = sales_df.groupby(["month", "store_name"])["total"].sum().reset_index()
    fig_line = px.line(
        monthly, x="month", y="total", color="store_name",
        labels={"month": "月", "total": "売上 (円)", "store_name": "店舗"},
        markers=True,
    )
    fig_line.update_layout(height=360, margin=dict(t=20, b=20), legend_title="")
    st.plotly_chart(fig_line, use_container_width=True)

with col_right:
    st.subheader("🏆 店舗別売上ランキング")
    ranking = sales_df.groupby("store_name")["total"].sum().sort_values().reset_index()
    fig_bar = px.bar(
        ranking, x="total", y="store_name", orientation="h",
        labels={"total": "売上 (円)", "store_name": "店舗"},
        color="total", color_continuous_scale="Blues",
    )
    fig_bar.update_layout(height=360, margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# ── 客単価 & アイドルタイム ──────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("💴 時間帯別 客単価")
    s_hour = sales_df[HOUR_COLS].sum()
    c_hour = cust_df[HOUR_COLS].sum()
    unit_by_hour = (s_hour / c_hour.replace(0, float("nan"))).fillna(0)
    fig_unit = px.bar(
        x=HOUR_COLS, y=unit_by_hour.values,
        labels={"x": "時間帯", "y": "客単価 (円)"},
        color=unit_by_hour.values, color_continuous_scale="Oranges",
    )
    fig_unit.update_layout(height=300, margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_unit, use_container_width=True)

with col_b:
    st.subheader("😴 アイドルタイム（売上¥0の割合）")
    idle = (sales_df[HOUR_COLS] == 0).mean() * 100
    fig_idle = px.bar(
        x=HOUR_COLS, y=idle.values,
        labels={"x": "時間帯", "y": "売上¥0の割合 (%)"},
        color=idle.values, color_continuous_scale="Reds",
    )
    fig_idle.update_layout(height=300, margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_idle, use_container_width=True)
    st.caption("割合が高い時間帯 ＝ SNS・タイムセール等の施策を打ちやすい時間帯")

st.divider()

# ── 詳細テーブル ──────────────────────────────────────
with st.expander("📋 店舗別詳細データ"):
    summary = (
        sales_df.groupby("store_name")["total"].sum().rename("売上合計").reset_index()
        .join(cust_df.groupby("store_name")["total"].sum().rename("客数合計"), on="store_name")
    )
    summary["客単価"] = (summary["売上合計"] / summary["客数合計"]).astype(int)
    summary["売上合計"] = summary["売上合計"].apply(lambda x: f"¥{x:,}")
    summary["客数合計"] = summary["客数合計"].apply(lambda x: f"{x:,}")
    summary["客単価"]   = summary["客単価"].apply(lambda x: f"¥{x:,}")
    st.dataframe(summary.rename(columns={"store_name": "店舗名"}), use_container_width=True)
