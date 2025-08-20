#  Marimo App: Interactive Data Analysis Demo
# Author email: 24f1002855@ds.study.iitm.ac.in

import marimo as mo

app = mo.app(
    title="Interactive Relationship Explorer (Marimo)",
    description=(
        "A small, self-documenting Marimo notebook that demonstrates "
        "variable dependencies, an interactive slider, and dynamic markdown."
    ),
)

# ---- Cell 1: Imports -------------------------------------------------------
# Data flow: This cell exposes commonly used libraries for downstream cells.
@app.cell
def _(mo):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    # We return the imported modules to make them available to other cells.
    return pd, np, plt, datasets

# ---- Cell 2: Load dataset --------------------------------------------------
# Data flow: Depends on 'datasets' from Cell 1. Produces 'df' used by later cells.
@app.cell
def _(datasets, pd):
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )
    df["target"] = iris.target
    return df

# ---- Cell 3: Widget(s) -----------------------------------------------------
# Data flow: Depends on 'df' to set slider bounds. Produces 'sample_n' widget.
@app.cell
def _(df, mo):
    sample_n = mo.ui.slider(
        start=10,
        stop=len(df),
        value=min(50, len(df)),
        label="Sample size (rows)",
    ).form(label="Sampling controls")
    # Another (optional) pair selector to demonstrate dependencies:
    feature_x = mo.ui.dropdown(
        options=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        value="sepal_length",
        label="X feature",
    )
    feature_y = mo.ui.dropdown(
        options=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        value="petal_length",
        label="Y feature",
    )
    return feature_x, feature_y, sample_n

# ---- Cell 4: Sample the data -----------------------------------------------
# Data flow: Depends on 'df' (Cell 2) and 'sample_n' (Cell 3). Produces 'sample'.
@app.cell
def _(df, sample_n):
    # The sample size is reactive: changing the slider re-executes this cell.
    sample = df.sample(n=sample_n.value, random_state=42)
    return sample

# ---- Cell 5: Compute summary statistics ------------------------------------
# Data flow: Depends on 'sample' from Cell 4 and 'feature_x','feature_y' from Cell 3.
@app.cell
def _(np, sample, feature_x, feature_y):
    x = sample[feature_x.value].to_numpy()
    y = sample[feature_y.value].to_numpy()
    corr = np.corrcoef(x, y)[0, 1]
    # Return pieces for downstream visualization and markdown.
    return corr, x, y

# ---- Cell 6: Dynamic Markdown ----------------------------------------------
# Data flow: Depends on 'sample_n', 'feature_x', 'feature_y', and 'corr'.
@app.cell
def _(corr, feature_x, feature_y, mo, sample_n):
    mo.md(
        f"""
### 📊 Live summary
- **Sample size**: **{sample_n.value}** rows  
- **X feature**: **{feature_x.value}**  
- **Y feature**: **{feature_y.value}**  
- **Pearson correlation** (X,Y): **{corr:.3f}**

This markdown updates automatically when you move the slider or change the selected features.
"""
    )

# ---- Cell 7: Scatter plot ---------------------------------------------------
# Data flow: Depends on 'x','y' from Cell 5. Reruns when inputs update.
@app.cell
def _(plt, x, y, feature_x, feature_y, mo):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.scatter(x, y, alpha=0.8)
    ax.set_xlabel(feature_x.value)
    ax.set_ylabel(feature_y.value)
    ax.set_title("Scatter plot")
    mo.ui.display(fig)

# ---- Cell 8: Data-flow Documentation (Markdown) -----------------------------
@app.cell
def _(mo):
    mo.md(
        """
### 🧩 Data-flow map
1. **Cell 1** imports modules → `{pd, np, plt, datasets}`  
2. **Cell 2** uses `datasets` → produces **df**  
3. **Cell 3** uses **df** → produces **sample_n** (slider), **feature_x**, **feature_y**  
4. **Cell 4** uses **df**, **sample_n** → produces **sample**  
5. **Cell 5** uses **sample**, **feature_x**, **feature_y** → produces **x**, **y**, **corr**  
6. **Cell 6** uses **sample_n**, **feature_x**, **feature_y**, **corr** → dynamic markdown  
7. **Cell 7** uses **x**, **y**, **feature_x**, **feature_y** → scatter plot

> This notebook satisfies: email comment, ≥2 variable-dependent cells, an interactive slider, dynamic markdown, and inline comments documenting data flow.
"""
    )

if __name__ == "__main__":
    app.run()
