"""
check_env.py — 第1回までに、これが動くことを確認しておいてください。

やること:
  1. 分析に使う3つのライブラリが入っているか
  2. 共通データが読めるか
  3. 回帰が1本走るか

使い方（このファイルがあるフォルダで）:
    python3 check_env.py

うまくいかないときは、**エラーの文面をそのままコピーして**持ってきてください。
「動きませんでした」だけでは、こちらも直せません。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ok = True


def report(label, passed, detail=""):
    global ok
    mark = "OK  " if passed else "NG  "
    print(f"  {mark}{label}" + (f"   {detail}" if detail else ""))
    if not passed:
        ok = False


print("=" * 62)
print("  秋学期 専門演習 / 環境チェック")
print("=" * 62)

# --- 1. Python 本体 ------------------------------------------------------
print("\n[1] Python")
v = sys.version_info
report(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 10),
       "" if v >= (3, 10) else "3.10 以上が必要です")

# --- 2. ライブラリ -------------------------------------------------------
print("\n[2] ライブラリ（この3つだけ使います）")
libs = {}
for name in ("pandas", "statsmodels", "matplotlib"):
    try:
        mod = __import__(name)
        libs[name] = mod
        report(name, True, getattr(mod, "__version__", ""))
    except ImportError as e:
        report(name, False, str(e))

# --- 3. 今いる場所 -------------------------------------------------------
# 春学期に最も多くの人を止めたのが、これです。
print("\n[3] 今どこにいるか")
import os
cwd = Path(os.getcwd()).resolve()
print(f"      作業ディレクトリ : {cwd}")
print(f"      このファイルの場所: {HERE}")
report("同じ場所から実行している", cwd == HERE,
       "" if cwd == HERE else "→ cd でこのファイルのある場所へ移動してから、もう一度実行してください")

# --- 4. 共通データ -------------------------------------------------------
print("\n[4] 共通データ")
csv = HERE / "data" / "country_panel.csv"
report("data/country_panel.csv がある", csv.exists())

if csv.exists() and "pandas" in libs:
    pd = libs["pandas"]
    d = pd.read_csv(csv)
    print(f"      行数     : {len(d)}")
    print(f"      国・地域 : {d['iso3'].nunique()}")
    print(f"      期間     : {d['year'].min()}-{d['year'].max()}")
    print(f"      列       : {list(d.columns)}")
    report("行数が 5,859 行", len(d) == 5859, f"実際は {len(d)} 行")

    # この4つを印字するまで「読めた」とは言いません。学期を通しての約束です。
    print("\n      欠測の数（列ごと）:")
    for c, n in d.isna().sum().items():
        if n:
            print(f"        {c:<16} {n}")

# --- 5. 回帰が1本走るか --------------------------------------------------
print("\n[5] 回帰が走るか")
if csv.exists() and {"pandas", "statsmodels"} <= libs.keys():
    try:
        import statsmodels.formula.api as smf
        s = libs["pandas"].read_csv(csv)[["milex_pct_gdp", "resource_rents"]].dropna()
        m = smf.ols("milex_pct_gdp ~ resource_rents", data=s).fit()
        b = m.params["resource_rents"]
        report("回帰が走った", True, f"係数 {b:+.3f} / n={int(m.nobs)}")
        report("係数が +0.047 前後", abs(b - 0.0466) < 0.002, f"実際は {b:+.4f}")
    except Exception as e:
        report("回帰が走った", False, f"{type(e).__name__}: {e}")

# --- 6. 図が描けるか -----------------------------------------------------
print("\n[6] 図が描けるか")
if "matplotlib" in libs:
    try:
        import matplotlib
        matplotlib.use("Agg")          # 画面を出さずにファイルへ保存する設定
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = HERE / "outputs" / "_check_env.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        report("図を保存できた", out.exists(), str(out.relative_to(HERE)))
        out.unlink(missing_ok=True)    # 確認用なので消しておく
    except Exception as e:
        report("図を保存できた", False, f"{type(e).__name__}: {e}")

# --- まとめ --------------------------------------------------------------
print("\n" + "=" * 62)
if ok:
    print("  すべて OK です。第1回はこのまま来てください。")
else:
    print("  NG があります。**エラーの文面をそのままコピーして**持ってきてください。")
    print("  直せなくて構いません。第1回の最初に、動いたかどうかだけ確認します。")
print("=" * 62)

sys.exit(0 if ok else 1)
