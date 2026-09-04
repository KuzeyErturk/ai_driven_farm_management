import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
POOLED_CSV = os.path.join(PROJECT, "data", "pooled",
                          "pooled_regional_crop_yield_weather_2004_2018.csv")
OUT_PATH = os.path.join(PROJECT, "plots", "european_yields_overview.png")

CROPS_PLOT = ["Wheat", "Barley", "Oats", "Oilseed_Rape"]
CROP_TITLE = {
    "Wheat": "Wheat",
    "Barley": "Barley",
    "Oats": "Oats",
    "Oilseed_Rape": "Oilseed Rape",
}

# Stable country order
COUNTRY_ORDER = ["UK", "France", "Germany", "Ireland",
                 "Netherlands", "Belgium", "Denmark"]

# Different colours for each country
COUNTRY_COLOURS = {
    "UK":          "#1f4e79",  
    "France":      "#c0392b",  
    "Germany":     "#2ca02c",  
    "Ireland":     "#8e44ad",  
    "Netherlands": "#e67e22",  
    "Belgium":     "#17becf",  
    "Denmark":     "#7f7f7f",  
}


def area_weighted_national(df_country_crop):
    g = df_country_crop.groupby("Year")[["Yield_t_per_ha", "Area_hectares"]].apply(
        lambda x: np.average(x["Yield_t_per_ha"],
                             weights=x["Area_hectares"])
    )
    return g.rename("Yield_t_per_ha").reset_index()


def national_yield_for_crop(df, country, crop):
    if crop == "Barley" and country != "UK":
        sub = df[(df.Country == country) &
                 (df.Crop.isin(["Spring_Barley", "Winter_Barley"]))].copy()
        if sub.empty:
            return None
        # Total production per region-year across the two barley types
        sub["Prod"] = sub["Yield_t_per_ha"] * sub["Area_hectares"]
        agg = (sub.groupby(["Year", "Region"])
                  .agg(Prod=("Prod", "sum"),
                       Area=("Area_hectares", "sum"))
                  .reset_index())
        agg["Yield_t_per_ha"] = agg["Prod"] / agg["Area"]
        agg = agg.rename(columns={"Area": "Area_hectares"})
        return area_weighted_national(agg)

    sub = df[(df.Country == country) & (df.Crop == crop)].copy()
    if sub.empty:
        return None
    return area_weighted_national(sub)


def main():
    if not os.path.exists(POOLED_CSV):
        sys.exit(f"ERROR: pooled CSV not found at {POOLED_CSV}")

    df = pd.read_csv(POOLED_CSV)
    print(f"Loaded {len(df):,} rows; countries = {sorted(df.Country.unique())}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()

    for ax, crop in zip(axes, CROPS_PLOT):
        for country in COUNTRY_ORDER:
            series = national_yield_for_crop(df, country, crop)
            if series is None or series.empty:
                continue
            lw = 2.4 if country == "UK" else 1.6
            ms = 6 if country == "UK" else 4
            ax.plot(series["Year"], series["Yield_t_per_ha"],
                    marker="o", linewidth=lw, markersize=ms,
                    color=COUNTRY_COLOURS[country], label=country)

        ax.set_title(CROP_TITLE[crop], fontsize=13, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Yield (t/ha)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(2003.5, 2016.5)

    # Single shared legend on the figure (not inside a subplot)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(COUNTRY_ORDER), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    fig.suptitle("European Crop Yields, 2004-2016",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
