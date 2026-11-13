import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="이 슈팅, 골이 될까?", page_icon="⚽", layout="wide")

GOAL_COLOR = "#d62728"
MISS_COLOR = "#9aa5b1"

DATA_URL = "https://raw.githubusercontent.com/ayoonjung/worldcup-shots/refs/heads/main/worldcup2022_shots.csv"
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)


raw = load_data()


COLUMN_INFO = pd.DataFrame(
    [
        ["match_id", "경기 고유 번호(64경기)"],
        ["team", "슈팅한 팀(32개 팀)"],
        ["player", "슈팅한 선수"],
        ["period", "경기 구간(1 전반, 2 후반, 3·4 연장 전·후반, 5 승부차기)"],
        ["minute", "슈팅한 시각(분)"],
        ["x", "슈팅 위치의 가로 좌표(0~120, 상대 골라인이 120)"],
        ["y", "슈팅 위치의 세로 좌표(0~80, 골대 가운데가 40)"],
        ["body_part", "슈팅한 몸 부위(4가지: Right Foot, Left Foot, Head, Other)"],
        ["shot_type", "슈팅 상황(4가지: Open Play, Free Kick, Penalty, Corner)"],
        ["under_pressure", "수비 압박을 받으며 쏜 슈팅이면 True, 아니면 빈칸"],
        ["outcome", "슈팅 결과(8가지: Goal, Saved, Off T, Blocked, Wayward, Post, Saved to Post, Saved Off Target)"],
        ["statsbomb_xg", "StatsBomb가 계산한 기대 득점(0~1)"],
    ],
    columns=["열 이름", "뜻"],
)


def page_data():
    st.header("1. 데이터 살펴보기")
    c1, c2, c3 = st.columns(3)
    c1.metric("경기 수", raw["match_id"].nunique())
    c2.metric("슈팅 수(행)", f"{len(raw):,}")
    c3.metric("열 수", raw.shape[1])

    st.subheader("데이터 미리보기")
    st.dataframe(raw.head(10), hide_index=True)

    st.subheader("열(속성) 설명")
    st.table(COLUMN_INFO, hide_index=True)

    st.subheader("슈팅 결과별 개수")
    counts = raw["outcome"].value_counts().reset_index()
    counts.columns = ["슈팅 결과", "개수"]
    fig = px.bar(counts, x="슈팅 결과", y="개수", text="개수")
    fig.update_layout(height=360)
    st.plotly_chart(fig)
    goal_n = int((raw["outcome"] == "Goal").sum())
    st.write(f"전체 슈팅 {len(raw):,}개 가운데 골은 {goal_n}개입니다(승부차기 포함).")


def prepare(data):
    d = data[data["period"] < 5]                      # 승부차기 제외
    d = d[d["shot_type"] == "Open Play"].copy()       # 페널티킥·프리킥·코너킥 직접 슈팅 제외
    d["under_pressure"] = d["under_pressure"].fillna(False).astype(bool)
    d["goal"] = (d["outcome"] == "Goal").astype(int)
    # 골대 가운데 (120, 40)까지 거리. 좌표 단위가 야드라서 0.9144를 곱해 m로 바꾼다
    d["distance"] = np.sqrt((120 - d["x"]) ** 2 + (40 - d["y"]) ** 2) * 0.9144
    # 양쪽 골대 기둥 (120, 36)·(120, 44)을 바라보는 두 방향 사이의 각
    d["angle"] = np.degrees(np.abs(np.arctan2(44 - d["y"], 120 - d["x"]) - np.arctan2(36 - d["y"], 120 - d["x"])))
    d["header"] = (d["body_part"] == "Head").astype(int)
    return d


shots = prepare(raw)
DIST_BINS = [0, 6, 12, 18, 24, 30, 200]
DIST_LABELS = ["0~6m", "6~12m", "12~18m", "18~24m", "24~30m", "30m 이상"]
shots["dist_group"] = pd.cut(shots["distance"], DIST_BINS, labels=DIST_LABELS, right=False)
shots["angle_group"] = np.where(shots["angle"] >= 20, "각도 20° 이상", "각도 20° 미만")
shots["part"] = np.where(shots["header"] == 1, "머리", "발·기타")


def pitch_figure(height=520):
    """골대가 위쪽에 오는 반쪽 경기장(가로 = y 좌표, 세로 = x 좌표)"""
    fig = go.Figure()
    line = dict(color="#7a7a7a", width=1.5)
    fig.add_shape(type="rect", x0=0, y0=60, x1=80, y1=120, line=line)       # 반쪽 경기장
    fig.add_shape(type="rect", x0=18, y0=102, x1=62, y1=120, line=line)     # 페널티 박스
    fig.add_shape(type="rect", x0=30, y0=114, x1=50, y1=120, line=line)     # 골 에어리어
    fig.add_shape(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color="black", width=5))  # 골대
    t = np.linspace(0, np.pi, 60)
    arc_x, arc_y = 40 + 10 * np.cos(t), 108 - 10 * np.sin(t)
    keep = arc_y < 102
    fig.add_trace(go.Scatter(x=arc_x[keep], y=arc_y[keep], mode="lines", line=line, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=40 + 10 * np.cos(t), y=60 + 10 * np.sin(t), mode="lines", line=line, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[40], y=[108], mode="markers", marker=dict(color="#7a7a7a", size=4), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(range=[-2, 82], visible=False)
    fig.update_yaxes(range=[58, 123], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=height, plot_bgcolor="white", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.04, x=0))
    return fig


def rate_chart(table, x, color=None, title=""):
    fig = px.bar(table, x=x, y="골 비율(%)", color=color, barmode="group", text="골 비율(%)",
                 hover_data=["슈팅 수", "골 수"], title=title,
                 labels={"dist_group": "골대까지 거리", "angle_group": "각도", "part": "몸 부위"})
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(height=380, yaxis_title="골 비율(%)")
    return fig


def goal_rate(df, by):
    t = df.groupby(by, observed=True)["goal"].agg(["count", "sum", "mean"]).reset_index()
    t.columns = list(t.columns[:-3]) + ["슈팅 수", "골 수", "골 비율(%)"]
    t["골 비율(%)"] = (t["골 비율(%)"] * 100).round(1)
    return t


def page_eda():
    st.header("2. 전처리와 시각화")

    st.subheader("전처리")
    shootout = int((raw["period"] == 5).sum())
    ingame = raw[raw["period"] < 5]
    steps = pd.DataFrame(
        [
            ["처음 데이터", len(raw)],
            [f"승부차기 {shootout}개 빼기", len(ingame)],
            ["페널티킥·프리킥·코너킥 직접 슈팅 빼기", len(shots)],
        ],
        columns=["단계", "남은 슈팅 수"],
    )
    removed = ingame["shot_type"].value_counts().drop("Open Play")
    c1, c2 = st.columns(2)
    with c1:
        st.table(steps, hide_index=True)
        st.write("뺀 세트피스: " + ", ".join(f"{k} {v}개" for k, v in removed.items()))
    with c2:
        missing = raw.isna().sum()
        st.table(missing[missing > 0].rename("빈칸 수").reset_index().rename(columns={"index": "열 이름"}), hide_index=True)
        st.write("under_pressure의 빈칸은 압박이 없었다는 뜻이라 False로 채웠습니다.")
    st.write(f"남은 슈팅 {len(shots):,}개 가운데 골은 {shots['goal'].sum()}개({shots['goal'].mean() * 100:.1f}%)입니다. "
             "새로 만든 열: goal(골이면 1), distance(골대까지 거리, m), angle(골대를 바라보는 각도, °), header(머리면 1)")
    st.dataframe(shots[["team", "player", "x", "y", "body_part", "outcome", "goal", "distance", "angle", "header"]].head(8).round(2),
                 hide_index=True)

    st.subheader("① 골이 된 슈팅은 어디에서 나왔을까?")
    team = st.selectbox("팀 고르기", ["전체"] + sorted(shots["team"].unique()))
    view = shots if team == "전체" else shots[shots["team"] == team]
    fig = pitch_figure()
    miss = view[view["goal"] == 0]
    goal = view[view["goal"] == 1]
    fig.add_trace(go.Scatter(x=miss["y"], y=miss["x"], mode="markers", name=f"골 아님 {len(miss)}개",
                             marker=dict(color=MISS_COLOR, size=6, opacity=0.6), text=miss["player"]))
    fig.add_trace(go.Scatter(x=goal["y"], y=goal["x"], mode="markers", name=f"골 {len(goal)}개",
                             marker=dict(color=GOAL_COLOR, size=9, line=dict(color="white", width=1)), text=goal["player"]))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("② 거리가 멀어지면 골 비율은?")
        st.plotly_chart(rate_chart(goal_rate(shots, "dist_group"), "dist_group").update_layout(xaxis_title="골대까지 거리"))
    with c2:
        st.subheader("③ 같은 거리에서 각도가 크면?")
        mid = shots[shots["dist_group"].isin(["6~12m", "12~18m"])]
        t = goal_rate(mid, ["dist_group", "angle_group"])
        st.plotly_chart(rate_chart(t, "dist_group", "angle_group").update_layout(xaxis_title="골대까지 거리(슈팅이 많은 6~18m)"))

    st.subheader("④ 머리 슈팅과 발 슈팅은 다를까?")
    c1, c2 = st.columns([2, 3])
    with c1:
        t = goal_rate(shots, "part").rename(columns={"part": "몸 부위"})
        t["골 비율(%)"] = t["골 비율(%)"].map("{:.1f}".format)
        t["평균 거리(m)"] = shots.groupby("part")["distance"].mean().map("{:.1f}".format).values
        st.table(t, hide_index=True)
        st.write("전체로 보면 골 비율이 거의 같지만, 머리 슈팅은 골대에 훨씬 가까이에서 나왔습니다.")
    with c2:
        near = shots[shots["dist_group"].isin(["0~6m", "6~12m", "12~18m"])]
        t = goal_rate(near, ["dist_group", "part"])
        st.plotly_chart(rate_chart(t, "dist_group", "part", "같은 거리 구간끼리 비교").update_layout(xaxis_title="골대까지 거리"))

    st.subheader("⑤ 한 경기에서 슈팅을 많이 한 팀이 골도 많이 넣었을까?")
    games = shots.groupby(["match_id", "team"]).agg(슈팅수=("goal", "size"), 골수=("goal", "sum")).reset_index()
    r = games["슈팅수"].corr(games["골수"])
    fig = px.scatter(games, x="슈팅수", y="골수", hover_data=["team"], opacity=0.6)
    fig.update_layout(height=380, xaxis_title="한 경기 슈팅 수(오픈 플레이)", yaxis_title="그 경기 골 수")
    st.plotly_chart(fig)
    st.write(f"팀·경기 {len(games)}개에서 슈팅 수와 골 수의 상관계수는 {r:.2f}입니다.")


FEATURES_A = ["distance"]
FEATURES_B = ["distance", "angle", "header"]
NAMES = {"distance": "거리(m)", "angle": "각도(°)", "header": "머리(1)"}

train_idx, test_idx = train_test_split(shots.index, test_size=0.2, random_state=42, stratify=shots["goal"])
train, test = shots.loc[train_idx], shots.loc[test_idx]
model_a = LogisticRegression().fit(train[FEATURES_A], train["goal"])
model_b = LogisticRegression().fit(train[FEATURES_B], train["goal"])


def front_angle(distance_m):
    """골대 정면에서 distance_m만큼 떨어진 곳의 각도"""
    gap = distance_m / 0.9144
    return np.degrees(2 * np.arctan2(4, gap))


def page_model():
    st.header("3. 모델 만들기(로지스틱 회귀 분류)")
    st.write("종속 변수는 goal(골이면 1, 아니면 0)입니다. 두 모델을 만들어 비교했습니다.")
    st.write("· 모델 A: 거리 하나만 넣은 모델  · 모델 B: 거리, 각도, 머리 여부를 넣은 모델")
    split = pd.DataFrame(
        [[name, f"{len(d):,}", int(d["goal"].sum()), f"{d['goal'].mean():.1%}"] for name, d in [("학습용", train), ("평가용", test)]],
        columns=["구분", "슈팅 수", "골 수", "골 비율"],
    )
    st.table(split, hide_index=True)
    st.write("슈팅을 8 : 2로 나누면서 두 묶음의 골 비율이 같아지게 했습니다(stratify).")

    st.subheader("모델이 배운 계수")
    rows = [["모델 A", NAMES[f], round(c, 4)] for f, c in zip(FEATURES_A, model_a.coef_[0])]
    rows += [["모델 B", NAMES[f], round(c, 4)] for f, c in zip(FEATURES_B, model_b.coef_[0])]
    rows += [["모델 A", "절편", round(model_a.intercept_[0], 4)], ["모델 B", "절편", round(model_b.intercept_[0], 4)]]
    st.table(pd.DataFrame(rows, columns=["모델", "변수", "계수"]), hide_index=True)
    st.write("계수가 음수면 그 값이 커질수록 골 확률이 낮아지고, 양수면 높아집니다.")

    st.subheader("골대 정면에서 거리에 따른 골 확률")
    dist = np.arange(2, 30.5, 0.5)
    ang = front_angle(dist)
    curve = pd.DataFrame({
        "거리(m)": np.concatenate([dist, dist, dist]),
        "골 확률": np.concatenate([
            model_a.predict_proba(pd.DataFrame({"distance": dist}))[:, 1],
            model_b.predict_proba(pd.DataFrame({"distance": dist, "angle": ang, "header": 0}))[:, 1],
            model_b.predict_proba(pd.DataFrame({"distance": dist, "angle": ang, "header": 1}))[:, 1],
        ]),
        "모델": ["모델 A"] * len(dist) + ["모델 B · 발"] * len(dist) + ["모델 B · 머리"] * len(dist),
    })
    fig = px.line(curve, x="거리(m)", y="골 확률", color="모델")
    fig.update_layout(height=420, yaxis_tickformat=".0%")
    st.plotly_chart(fig)


def evaluate(model, features, threshold):
    prob = model.predict_proba(test[features])[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test["goal"], pred, labels=[0, 1]).ravel()
    return {
        "정확도": accuracy_score(test["goal"], pred),
        "정밀도": precision_score(test["goal"], pred, zero_division=0),
        "재현율": recall_score(test["goal"], pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "prob": prob,
    }


def matrix_table(e):
    return pd.DataFrame([[e["tn"], e["fp"]], [e["fn"], e["tp"]]],
                        index=["실제 골 아님", "실제 골"], columns=["예측 골 아님", "예측 골"])


def pct(v):
    return f"{v:.1%}"


def precision_text(e, dash="\\-"):
    """골이라고 예측한 슈팅이 하나도 없으면 정밀도를 계산할 수 없어 -로 쓴다(표 칸에서는 \\-로 써야 보인다)"""
    return dash if e["tp"] + e["fp"] == 0 else pct(e["정밀도"])


def page_eval():
    st.header("4. 모델 평가")
    baseline = (test["goal"] == 0).mean()
    a, b = evaluate(model_a, FEATURES_A, 0.5), evaluate(model_b, FEATURES_B, 0.5)
    st.subheader(f"기준값 0.5로 골/골 아님을 나눴을 때(평가용 슈팅 {len(test)}개)")
    table = pd.DataFrame(
        [
            ["모두 '골 아님'으로 찍기", pct(baseline), "\\-", pct(0)],
            ["모델 A(거리)", pct(a["정확도"]), precision_text(a), pct(a["재현율"])],
            ["모델 B(거리·각도·머리)", pct(b["정확도"]), precision_text(b), pct(b["재현율"])],
        ],
        columns=["방법", "정확도", "정밀도", "재현율"],
    )
    st.table(table, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("모델 A 혼동행렬")
        st.table(matrix_table(a))
    with c2:
        st.write("모델 B 혼동행렬")
        st.table(matrix_table(b))

    st.subheader("기준값을 바꿔 보기(모델 B)")
    th = st.slider("골로 판단하는 확률 기준값", 0.05, 0.9, 0.5, 0.05)
    e = evaluate(model_b, FEATURES_B, th)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    c1.metric("정확도", pct(e["정확도"]))
    c2.metric("정밀도", precision_text(e, dash="-"))
    c3.metric("재현율", pct(e["재현율"]))
    with c4:
        st.table(matrix_table(e))

    rows = []
    for t in [0.5, 0.3, 0.15]:
        for name, m, f in [("모델 A", model_a, FEATURES_A), ("모델 B", model_b, FEATURES_B)]:
            r = evaluate(m, f, t)
            rows.append([str(t), name, pct(r["정확도"]), precision_text(r), pct(r["재현율"]), r["tp"], r["tp"] + r["fp"]])
    st.table(pd.DataFrame(rows, columns=["기준값", "모델", "정확도", "정밀도", "재현율", "맞힌 골", "골이라고 한 슈팅"]), hide_index=True)

    st.subheader("확률로 보기")
    rows = []
    for name, e in [("모델 A", a), ("모델 B", b)]:
        p = e["prob"]
        rows.append([name, f"{p.sum():.2f}", int(test["goal"].sum()),
                     f"{p[test['goal'].values == 1].mean():.3f}", f"{p[test['goal'].values == 0].mean():.3f}"])
    st.table(pd.DataFrame(rows, columns=["모델", "확률 합", "실제 골 수", "골이 된 슈팅의 평균 확률", "골이 안 된 슈팅의 평균 확률"]),
             hide_index=True)


KOREA = "South Korea"
OPPONENTS = {"Uruguay": "조별리그 1차전", "Ghana": "조별리그 2차전", "Portugal": "조별리그 3차전", "Brazil": "16강"}
KOREAN_NAMES = {"Uruguay": "우루과이", "Ghana": "가나", "Portugal": "포르투갈", "Brazil": "브라질"}
shots["prob"] = model_b.predict_proba(shots[FEATURES_B])[:, 1]


def goal_probability(forward_m, side_m, is_header):
    x = 120 - forward_m / 0.9144
    y = 40 + side_m / 0.9144
    distance = np.sqrt((120 - x) ** 2 + (40 - y) ** 2) * 0.9144
    angle = np.degrees(abs(np.arctan2(44 - y, 120 - x) - np.arctan2(36 - y, 120 - x)))
    X = pd.DataFrame({"distance": [distance], "angle": [angle], "header": [int(is_header)]})
    return model_b.predict_proba(X)[0, 1], x, y, distance, angle

def page_calc():
    st.header("5. 골 확률 계산기와 한국 경기 분석")
    st.subheader("골 확률 계산기(모델 B)")
    c1, c2 = st.columns([2, 3])
    with c1:
        forward = st.slider("골라인에서 떨어진 거리(m)", 0.5, 35.0, 11.0, 0.5)
        side = st.slider("골대 가운데에서 옆으로 떨어진 거리(m)", 0.0, 30.0, 0.0, 0.5)
        part = st.radio("슈팅한 몸 부위", ["발", "머리"], horizontal=True)
        p, x, y, dist, ang = goal_probability(forward, side, part == "머리")
        st.metric("골이 될 확률", f"{p:.1%}")
        st.write(f"골대까지 거리 {dist:.1f} m, 골대를 바라보는 각도 {ang:.1f}°")
    with c2:
        fig = pitch_figure(height=430)
        fig.add_trace(go.Scatter(x=[y], y=[x], mode="markers", name="슈팅 위치",
                                 marker=dict(color=GOAL_COLOR, size=14, symbol="x")))
        st.plotly_chart(fig)

    st.subheader("한국 대표팀 4경기: 슈팅 수와 기대 득점(확률 합)")
    korea_ids = raw.loc[raw["team"] == KOREA, "match_id"].unique()
    match_of = {opp: raw[(raw["team"] == opp) & raw["match_id"].isin(korea_ids)]["match_id"].iloc[0] for opp in OPPONENTS}
    rows = []
    for opp, label in OPPONENTS.items():
        mid = match_of[opp]
        ingame = raw[(raw["match_id"] == mid) & (raw["period"] < 5)]
        score = f"{(ingame[ingame['team'] == KOREA]['outcome'] == 'Goal').sum()} : {(ingame[ingame['team'] == opp]['outcome'] == 'Goal').sum()}"
        for team in [KOREA, opp]:
            g = shots[(shots["match_id"] == mid) & (shots["team"] == team)]
            rows.append([f"{label} {KOREAN_NAMES[opp]}", score, "한국" if team == KOREA else KOREAN_NAMES[opp], len(g), int(g["goal"].sum()),
                         f"{g['prob'].sum():.2f}", f"{g['prob'].mean():.3f}"])
    st.table(pd.DataFrame(rows, columns=["경기", "점수(한국 : 상대)", "팀", "슈팅 수", "골", "기대 득점(확률 합)", "슈팅 하나당 평균 확률"]),
             hide_index=True)
    st.caption("슈팅 수·골·기대 득점은 오픈 플레이 슈팅만 센 값입니다. 점수는 페널티킥 등을 포함한 실제 점수입니다.")

    g = shots[shots["match_id"] == match_of["Ghana"]]
    fig = pitch_figure(height=470)
    for team, color in [(KOREA, "#1f77b4"), ("Ghana", "#ff7f0e")]:
        t = g[g["team"] == team]
        fig.add_trace(go.Scatter(x=t["y"], y=t["x"], mode="markers", name="한국" if team == KOREA else "가나",
                                 marker=dict(color=color, size=8 + 40 * t["prob"], opacity=0.75,
                                             line=dict(color=np.where(t["goal"] == 1, "black", "white"), width=2)),
                                 text=t["player"] + "  골 확률 " + (t["prob"] * 100).round(1).astype(str) + "%"))
    fig.update_layout(title="한국 대 가나 슈팅 지도(원이 클수록 골 확률이 높음, 검은 테두리 = 골)")
    st.plotly_chart(fig)

    st.subheader("한 경기 슈팅 수와 기대 득점 가운데 어느 쪽이 실제 골 수와 더 잘 맞을까?")
    games = shots.groupby(["match_id", "team"]).agg(슈팅수=("goal", "size"), 기대득점=("prob", "sum"), 골수=("goal", "sum")).reset_index()
    compare = pd.DataFrame(
        [["슈팅 수", f"{games['슈팅수'].corr(games['골수']):.2f}"], ["기대 득점(확률 합)", f"{games['기대득점'].corr(games['골수']):.2f}"]],
        columns=["한 경기에서 센 값", "실제 골 수와의 상관계수"],
    )
    st.table(compare, hide_index=True)
    st.write(f"팀·경기 {len(games)}개로 계산했습니다.")

    st.subheader("StatsBomb 기대 득점과 비교")
    r = shots["prob"].corr(shots["statsbomb_xg"])
    fig = px.scatter(shots, x="statsbomb_xg", y="prob", opacity=0.4)
    fig.update_layout(height=380, xaxis_title="StatsBomb 기대 득점", yaxis_title="모델 B 골 확률")
    st.plotly_chart(fig)
    st.write(f"슈팅 {len(shots):,}개에서 두 값의 상관계수는 {r:.2f}입니다.")

st.title("이 슈팅, 골이 될까?")
st.caption("2022 카타르 월드컵 64경기 슈팅 데이터로 골이 될 확률 예측하기 · 데이터: StatsBomb Open Data")
pages = [
    st.Page(page_data, title="1. 데이터 살펴보기", url_path="data", default=True),
    st.Page(page_eda, title="2. 전처리와 시각화", url_path="eda"),
    st.Page(page_model, title="3. 모델 만들기", url_path="model"),
    st.Page(page_eval, title="4. 모델 평가", url_path="evaluate"),
    st.Page(page_calc, title="5. 골 확률 계산기", url_path="calculator"),
]
st.navigation(pages).run()
