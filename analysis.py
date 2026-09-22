"""
analysis.py — 分析はこの1本に書きます。

【約束】
  1. 上から下まで一気に走ること。途中で手作業を挟まない
  2. パスは相対で書く（この 3 行目の HERE を使う）
  3. 図と表は outputs/ に保存する
  4. 主役の回帰は1本。モデルは最大2列（単純なもの／統制または固定効果）

「上から下まで一気に走る」は加点対象です（§4）。必須ではありません。
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")            # 画面を出さずにファイルへ保存する
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. データを読む
# ============================================================
# 自分のデータを使う人は、ここを data/public/自分のファイル.csv に変えます
d = pd.read_csv(HERE / "data" / "country_panel.csv")

# --- 読めたことの確認（毎回必ず印字する。これを見るまで終わっていません） ---
print("=== 読み込み ===")
print("行数      :", len(d))
print("列        :", d.columns.tolist())
print("期間      :", d["year"].min(), "-", d["year"].max())
print("欠測      :")
print(d.isna().sum().to_string())

# ============================================================
# 2. 分析に使う変数を決める
# ============================================================
Y = "milex_pct_gdp"        # 説明されるもの
X = "resource_rents"       # 説明したいもの（主役）
UNIT = "iso3"              # 分析単位（固定効果に使う）

s = d[[UNIT, "year", Y, X]].dropna()

# --- dropna() で何行消えたかを必ず数える（黙って捨てさせない） ---
print("\n=== 欠測の除外 ===")
print(f"除外前: {len(d)} 行 / 除外後: {len(s)} 行 / 消えた: {len(d) - len(s)} 行")
print(f"国の数: {d[UNIT].nunique()} → {s[UNIT].nunique()} "
      f"（{d[UNIT].nunique() - s[UNIT].nunique()} か国が丸ごと消えた）")

# ============================================================
# 3. 表1（記述統計）
# ============================================================
tab1 = s[[Y, X]].describe().T[["count", "mean", "std", "min", "50%", "max"]]
print("\n=== 表1 記述統計 ===")
print(tab1.round(3).to_string())
tab1.round(3).to_csv(OUT / "table1_descriptive.csv", encoding="utf-8-sig")

# ============================================================
# 4. 図1（散布図）
# ============================================================
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(s[X], s[Y], s=10, alpha=.4)
ax.set_xlabel(X)
ax.set_ylabel(Y)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "figure1.png", dpi=150)
plt.close(fig)
print(f"\n図1を保存しました: outputs/figure1.png")

# ============================================================
# 5. 主役の回帰（最大2列）
# ============================================================
print("\n=== 回帰(1) 単純 ===")
m1 = smf.ols(f"{Y} ~ {X}", data=s).fit()
print(m1.summary())

print("\n=== 回帰(2) 個体ダミー（固定効果） ===")
# 自分のデータがパネルでない人は、この回帰(2)を消して構いません。
# 消す場合は「なぜ使わないのか」を paper.md の 4.2 に1文で書いてください。
m2 = smf.ols(f"{Y} ~ {X} + C({UNIT})", data=s).fit(
    cov_type="cluster", cov_kwds={"groups": s[UNIT]}
)
print(m2.summary().tables[0])
print(m2.summary().tables[1].as_text()[:400], "\n  …（個体ダミーの行は省略）")

# --- 見る数字は4つだけ ---
print("\n=== 読む数字（この4つだけ） ===")
for name, m in (("単純", m1), ("固定効果", m2)):
    ci = m.conf_int().loc[X]
    print(f"  {name:<6} n={int(m.nobs):>5}  coef={m.params[X]:+.4f}  "
          f"95%CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]  p={m.pvalues[X]:.3g}")

# --- 重複した列がないかの確認（第6回で扱います） ---
print("\n=== 列の重複チェック ===")
for name, m in (("単純", m1), ("固定効果", m2)):
    rank, ncol = int(m.df_model) + 1, m.model.exog.shape[1]
    flag = "一致" if rank == ncol else "★不一致 → 重複した列があります"
    print(f"  {name:<6} rank={rank:>4}  列数={ncol:>4}  {flag}")

# ============================================================
# 6. 出力を保存する（第9回の提出物）
# ============================================================
(OUT / "summary_model1.txt").write_text(m1.summary().as_text(), encoding="utf-8")
(OUT / "summary_model2.txt").write_text(m2.summary().as_text(), encoding="utf-8")
print("\noutputs/ に summary を保存しました。")

# ============================================================
# 7. 頑健性チェック（加点・第9回）
# ============================================================
# Xが大きい上位3つの「単位」を丸ごと外すと、係数はどう変わるか。
#
# ★ 落とすのは「行」ではなく「単位（国）」です。
#   パネルデータで上位3行だけ落としても、3/3965 なので何も変わりません。
#   結果を動かすのは「その国が全期間ずっと外れている」ことのほうです。
top3 = s.groupby(UNIT)[X].mean().nlargest(3)
s2 = s[~s[UNIT].isin(top3.index)]
m3 = smf.ols(f"{Y} ~ {X}", data=s2).fit()

print("\n=== 頑健性チェック（Xが大きい上位3か国を丸ごと外す） ===")
print(f"  外した単位  : {', '.join(f'{k}({v:.1f})' for k, v in top3.items())}")
print(f"  全体        : coef={m1.params[X]:+.4f}  n={int(m1.nobs)}  "
      f"単位数={s[UNIT].nunique()}")
print(f"  上位3を除外 : coef={m3.params[X]:+.4f}  n={int(m3.nobs)}  "
      f"単位数={s2[UNIT].nunique()}")
print(f"  → 係数は {m3.params[X] / m1.params[X]:.2f} 倍になりました")
print("\n  paper.md の考察に、この1行を書きます:")
print(f"  「Xが大きい上位3単位を除外しても係数は {m3.params[X]:+.3f} "
      f"（除外前 {m1.params[X]:+.3f}）であり、結論は変わらなかった／変わった」")
